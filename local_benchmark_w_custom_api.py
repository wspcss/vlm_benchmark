#!/usr/bin/env python3
"""
Custom-API VLM Benchmark Script

Like ``local_benchmark.py`` but runs inference against any OpenAI-compatible
chat/completions endpoint (e.g. an mlx_lm / vLLM / sglang server hosting a
quantized VLM) instead of loading the model in-process via mlx_vlm. This keeps
the script runnable on non-Apple-Silicon boxes (Linux/Windows) that can't
import mlx_vlm — it only needs HTTP + the OpenRouter key for the judge.

The LLM-judge scoring (OpenRouter), response cleanup, HTML report, and overall
flow are identical to ``local_benchmark.py``; only the inference path is
swapped: a POST to ``{base_url}/v1/chat/completions`` with an image data URI.

Usage:
    python3 local_benchmark_w_custom_api.py \\
        --base-url http://192.168.1.10:8000 \\
        --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit

Defaults for ``--base-url`` / ``--api-key`` may also be supplied via the
``VLM_API_BASE_URL`` / ``VLM_API_KEY`` env vars (sourced from ``.env`` by
``run_local_benchmark_w_custom_api.sh``). CLI flags override env vars.
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

from PIL import Image

# --- Judge (OpenRouter) -----------------------------------------------------
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
# Same judge as local_benchmark.py
JUDGE_MODEL = "deepseek/deepseek-v4-flash"

# --- Test suite / generation ------------------------------------------------
TEST_CASES_FILE = "test_case.json"
MAX_TOKENS = 512

# Hard cap on an image's longest edge before sending it to the server. Driven
# by the model family's image-processor prefill budget (see progress.md).
# Qwen2.5-VL tolerates 10k+ prefill tokens, so 1920 keeps native 1920x1080/1200
# screenshots untouched while only downscaling anything larger. If you point
# this at a Qwen3-VL server (1280px ceiling), pass --max-image-side 1280.
MAX_IMAGE_SIDE = 1920


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------

def load_image(image_path, max_side=MAX_IMAGE_SIDE):
    """Open an image and downscale so its longest edge is <= max_side."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return img


def image_to_data_uri(image_path, max_side=MAX_IMAGE_SIDE):
    """Open + downscale an image and return a base64 data URI.

    Re-encodes as PNG (lossless) so screenshot text stays sharp for OCR / state
    reading tasks; the LAN upload cost of PNG is negligible vs. a cloud API.
    """
    img = load_image(image_path, max_side)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def encode_image_to_base64(image_path):
    """Read an image file verbatim and return a base64 data URI (for HTML embed)."""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(ext, "image/png")
    return f"data:{mime_type};base64,{encoded}"


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def _post_json(url, payload, headers, timeout):
    """POST JSON and return (parsed_body, elapsed_seconds)."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers=headers, method="POST"
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            response_data = json.loads(body)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"API error {e.code}: {error_body}")
    elapsed = time.time() - start
    return response_data, elapsed


def call_openrouter(model, messages, max_tokens):
    """Call the OpenRouter chat completions API (used for judge scoring)."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    return _post_json(OPENROUTER_API_URL, payload, headers, timeout=120)


def call_vlm_api(base_url, api_key, model, image_path, question,
                 max_tokens, max_image_side):
    """Run a single vision inference against the custom OpenAI-compatible API.

    Returns a dict mirroring run_local_inference's shape (prompt_tokens,
    generation_tokens, generation_tps, time_seconds) so the rest of the
    benchmark pipeline is unchanged. ``peak_memory_gb`` is N/A for a remote
    server (we cannot observe its memory) and is set to None.
    """
    data_uri = image_to_data_uri(image_path, max_image_side)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": question},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        # Suppress chain-of-thought at the source, mirroring the in-process
        # enable_thinking=False passed by local_benchmark.py. mlx_lm's OpenAI
        # server forwards chat_template_kwargs into apply_chat_template; other
        # servers that don't recognize it will simply ignore the extra field.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    response_data, elapsed = _post_json(endpoint, payload, headers, timeout=300)

    choices = response_data.get("choices", [])
    text = ""
    if choices:
        text = (choices[0].get("message", {}).get("content") or "").strip()

    usage = response_data.get("usage", {}) or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    generation_tokens = usage.get("completion_tokens", 0)
    tps = (generation_tokens / elapsed) if elapsed > 0 else 0.0

    return {
        "response_text": text,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "generation_tps": tps,
        "peak_memory_gb": None,  # not observable for a remote server
        "time_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Judge + response cleanup (identical to local_benchmark.py)
# ---------------------------------------------------------------------------

def judge_response(question, grounded_answer, model_response):
    """Use an LLM judge to score the model's response on a scale of 0-10."""
    judge_prompt = (
        f"Given this question:\n\"{question}\"\n\n"
        f"The expected answer is:\n\"{grounded_answer}\"\n\n"
        f"The model responded:\n\"{model_response}\"\n\n"
        f"Rate how accurately the model's response matches the expected answer "
        f"on a scale of 0 to 10, where 0 is completely wrong and 10 is when all points in the expected answer are covered. "
        f"Reply with only a single number."
    )

    messages = [{"role": "user", "content": judge_prompt}]

    try:
        response_data, _ = call_openrouter(JUDGE_MODEL, messages, max_tokens=512)
        choices = response_data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            text = (msg.get("content") or "").strip()
            if not text:
                reasoning = (msg.get("reasoning") or "").strip()
                if reasoning:
                    text = reasoning
            num_match = re.search(r'\b(10|\d)\b', text)
            if num_match:
                return int(num_match.group())
        return None
    except Exception as e:
        print(f"    ⚠️  Judge API call failed: {e}")
        return None


def strip_box_markers(text):
    """Remove GLM-4.6V's final-answer box markers from a response.

    GLM-4.6V wraps its final answer in the special tokens
    ``<|begin_of_box|>...<|end_of_box|>``. These are emitted by the model
    during generation (not by the chat template), so there is no prompt-side
    flag to suppress them — they must be stripped after the fact. Prefers the
    boxed content if present, otherwise removes any stray marker. A no-op for
    models that don't emit the markers (e.g. Qwen2.5-VL).
    """
    if not text:
        return text

    box = re.search(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", text, re.DOTALL)
    if box:
        text = box.group(1)
    else:
        text = text.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")

    return text.strip()


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def sanitize_model_name(model_name):
    return model_name.replace("/", "_").strip()


def run_benchmark(model_name, base_url, api_key, tests, max_tokens, max_image_side):
    """Run benchmark for a single model served by the custom API."""
    print(f"\n{'=' * 70}")
    print(f"CUSTOM-API VLM BENCHMARK: {model_name}")
    print(f"{'=' * 70}")

    results = []
    total_start = time.time()
    total_input_tokens = 0
    total_output_tokens = 0

    for i, case in enumerate(tests, 1):
        question = case["question"]
        grounded = case.get("grounded_answer", "")
        image_path = case.get("image", "")
        if not image_path:
            print(f"\n  ❌ No image specified for test case {i}")
            continue
        if not os.path.exists(image_path):
            print(f"\n  ❌ Image not found: {image_path}")
            continue

        category = case.get("category", "")
        print(f"\n  Image: {image_path}")
        if category:
            print(f"  Category: {category}")

        try:
            inference = call_vlm_api(
                base_url, api_key, model_name, image_path, question,
                max_tokens, max_image_side,
            )
        except Exception as e:
            print(f"\n  ❌ API inference failed: {e}")
            results.append({
                "question": question,
                "model_response": f"ERROR: {e}",
                "grounded_answer": grounded,
                "image": image_path,
                "category": category,
                "accuracy_score": None,
                "time_seconds": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "tokens_per_second": 0,
                "peak_memory_gb": None,
            })
            continue

        response_text = strip_box_markers(inference["response_text"])
        input_tokens = inference["prompt_tokens"]
        output_tokens = inference["generation_tokens"]
        tps = inference["generation_tps"]
        elapsed = inference["time_seconds"]

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        if grounded.strip():
            print(f"  🤖 Asking judge ({JUDGE_MODEL})...")
            accuracy_score = judge_response(question, grounded, response_text)
            if accuracy_score is not None:
                if accuracy_score >= 7:
                    status = f"✅ {accuracy_score}/10"
                elif accuracy_score >= 4:
                    status = f"⚠️  {accuracy_score}/10"
                else:
                    status = f"❌ {accuracy_score}/10"
            else:
                status = "📝 JUDGE FAILED"
        else:
            accuracy_score = None
            status = "📝 INFO (no grounded answer)"

        print(f"\n[{i}/{len(tests)}] {status}")
        print(f"  Q:      {question[:100]}...")
        print(f"  Model:  {response_text[:400]}")
        if grounded.strip():
            print(f"  Truth:  {grounded}")
        if accuracy_score is not None:
            print(f"  Judge:  {accuracy_score}/10")
        print(f"  Time:   {elapsed:.2f}s | Input: {input_tokens} | Output: {output_tokens} | TPS: {tps:.1f}")

        results.append({
            "question": question,
            "model_response": response_text,
            "grounded_answer": grounded,
            "image": image_path,
            "category": category,
            "accuracy_score": accuracy_score,
            "time_seconds": round(elapsed, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_second": round(tps, 1),
            "peak_memory_gb": None,
        })

    total_elapsed = time.time() - total_start

    valid_results = [r for r in results if r["time_seconds"] > 0]
    avg_tps = (
        sum(r["tokens_per_second"] for r in valid_results) / len(valid_results)
        if valid_results else 0
    )
    avg_time = (
        sum(r["time_seconds"] for r in valid_results) / len(valid_results)
        if valid_results else 0
    )

    scored_results = [r for r in results if r["accuracy_score"] is not None]
    total_scored = len(scored_results)
    avg_score = (
        sum(r["accuracy_score"] for r in scored_results) / total_scored
        if total_scored > 0 else None
    )

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Model:            {model_name}")
    print(f"Endpoint:         {base_url}")
    print(f"Test cases:       {len(results)}")
    if avg_score is not None:
        print(f"Avg judge score:  {avg_score:.1f}/10 ({total_scored} scored)")
    print(f"Total time:       {total_elapsed:.2f}s")
    print(f"Avg per case:     {avg_time:.2f}s")
    print(f"Avg TPS:          {avg_tps:.1f}")
    print(f"Total tokens in:  {total_input_tokens}")
    print(f"Total tokens out: {total_output_tokens}")

    summary = {
        "model": model_name,
        "endpoint": base_url,
        "total_cases": len(results),
        "scored_cases": total_scored,
        "avg_judge_score": round(avg_score, 1) if avg_score is not None else None,
        "total_time_s": round(total_elapsed, 2),
        "avg_per_case_s": round(avg_time, 2),
        "avg_tokens_per_second": round(avg_tps, 1),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "results": results,
    }
    return summary


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_html_report(summaries, output_path):
    """Generate an HTML report from benchmark results (single custom-API model)."""
    import html as html_lib

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Model comparison table
    comparison_rows = ""
    for s in summaries:
        avg_judge = s.get("avg_judge_score")
        if avg_judge is not None:
            if avg_judge >= 7:
                score_badge = f'<span style="color:#16a34a;font-weight:bold;">{avg_judge:.1f}/10</span>'
            elif avg_judge >= 4:
                score_badge = f'<span style="color:#d97706;font-weight:bold;">{avg_judge:.1f}/10</span>'
            else:
                score_badge = f'<span style="color:#dc2626;font-weight:bold;">{avg_judge:.1f}/10</span>'
        else:
            score_badge = '<span style="color:#999;">N/A</span>'

        comparison_rows += f"""
        <tr>
          <td>{html_lib.escape(s['model'])}</td>
          <td>{html_lib.escape(s.get('endpoint', ''))}</td>
          <td>{s['avg_tokens_per_second']:.1f}</td>
          <td>{s['avg_per_case_s']:.2f}s</td>
          <td>{score_badge}</td>
          <td>{s['total_input_tokens']}</td>
          <td>{s['total_output_tokens']}</td>
        </tr>"""

    # Per-test-case sections
    max_cases = max((len(s.get("results", [])) for s in summaries), default=0)
    test_case_sections = ""
    for case_idx in range(max_cases):
        case_info = None
        for s in summaries:
            results = s.get("results", [])
            if case_idx < len(results):
                case_info = results[case_idx]
                break
        if not case_info:
            continue

        question = case_info.get("question", "")
        grounded = case_info.get("grounded_answer", "")
        img_path = case_info.get("image", "")
        category = case_info.get("category", "")

        image_html = ""
        if img_path and os.path.exists(img_path):
            img_b64 = encode_image_to_base64(img_path)
            image_html = f'<img src="{img_b64}" alt="{html_lib.escape(img_path)}" style="max-height:300px;border-radius:8px;border:1px solid #e2e8f0;" />'

        comparison_rows_html = ""
        for s in summaries:
            results = s.get("results", [])
            if case_idx >= len(results):
                continue
            r = results[case_idx]

            score = r.get("accuracy_score")
            if score is not None:
                if score >= 7:
                    score_cell = f'<td style="color:#16a34a;font-weight:bold;">{score}/10</td>'
                elif score >= 4:
                    score_cell = f'<td style="color:#d97706;font-weight:bold;">{score}/10</td>'
                else:
                    score_cell = f'<td style="color:#dc2626;font-weight:bold;">{score}/10</td>'
            else:
                score_cell = '<td style="color:#999;">N/A</td>'

            response_escaped = html_lib.escape(r.get("model_response", ""))
            comparison_rows_html += f"""
            <tr>
              <td>{html_lib.escape(s['model'])}</td>
              <td style="max-width:600px;">{response_escaped}</td>
              {score_cell}
              <td>{r.get('time_seconds', 0):.2f}s</td>
              <td>{r.get('tokens_per_second', 0):.1f}</td>
            </tr>"""

        category_badge = f'<span style="background:#e0e7ff;color:#3730a3;padding:0.2rem 0.6rem;border-radius:9999px;font-size:0.8rem;font-weight:600;">{html_lib.escape(category)}</span>' if category else ''
        test_case_sections += f"""
    <div class="test-case-section">
      <h2>Test Case {case_idx + 1} {category_badge}</h2>
      <div class="case-info">
        {image_html}
        <div class="case-details">
          <p><strong>Question:</strong> {html_lib.escape(question)}</p>
          <p><strong>Expected Answer:</strong> {html_lib.escape(grounded)}</p>
        </div>
      </div>
      <table>
        <thead>
          <tr><th>Model</th><th>Response</th><th>Score</th><th>Time</th><th>TPS</th></tr>
        </thead>
        <tbody>{comparison_rows_html}
        </tbody>
      </table>
    </div>"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Custom-API VLM Benchmark Results - {timestamp}</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8fafc; color: #1e293b; padding: 2rem; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; color: #0f172a; }}
    .timestamp {{ color: #64748b; margin-bottom: 2rem; font-size: 0.95rem; }}
    .comparison {{ background: white; border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .comparison h2 {{ font-size: 1.2rem; margin-bottom: 1rem; color: #334155; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th {{ background: #f1f5f9; padding: 0.75rem 1rem; text-align: left; font-weight: 600; color: #475569; border-bottom: 2px solid #e2e8f0; }}
    td {{ padding: 0.65rem 1rem; border-bottom: 1px solid #f1f5f9; vertical-align: top; }}
    tr:hover {{ background: #f8fafc; }}
    .test-case-section {{ background: white; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .test-case-section h2 {{ font-size: 1.2rem; margin-bottom: 1rem; color: #334155; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; }}
    .case-info {{ display: flex; gap: 1.5rem; margin-bottom: 1.5rem; align-items: flex-start; flex-wrap: wrap; }}
    .case-details {{ flex: 1; min-width: 300px; }}
    .case-details p {{ margin-bottom: 0.5rem; line-height: 1.5; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>🖥️ Custom-API VLM Benchmark Results</h1>
    <p class="timestamp">Generated: {timestamp} | Models: {len(summaries)} | Judge: {html_lib.escape(JUDGE_MODEL)}</p>

    <div class="comparison">
      <h2>Model Comparison</h2>
      <table>
        <thead>
          <tr><th>Model</th><th>Endpoint</th><th>Avg TPS</th><th>Avg/Case</th><th>Avg Score</th><th>Tokens In</th><th>Tokens Out</th></tr>
        </thead>
        <tbody>{comparison_rows}
        </tbody>
      </table>
    </div>
    {test_case_sections}
  </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_model(base_url, api_key):
    """Query {base_url}/v1/models and return the first model id, or None."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/models", headers=headers, method="GET"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = data.get("data", []) or []
        if models:
            return models[0].get("id")
    except Exception as e:
        print(f"  ⚠️  Could not query /v1/models: {e}")
    return None


def main():
    global OPENROUTER_API_KEY, JUDGE_MODEL

    parser = argparse.ArgumentParser(description="Custom-API VLM Benchmark")
    parser.add_argument("--base-url", type=str,
                        default=os.environ.get("VLM_API_BASE_URL", ""),
                        help="Base URL of the OpenAI-compatible server "
                             "(e.g. http://192.168.1.10:8000). "
                             "Default: env VLM_API_BASE_URL.")
    parser.add_argument("--model", type=str, default=None,
                        help="Model id to use. If omitted, the first model "
                             "reported by GET /v1/models is used.")
    parser.add_argument("--api-key", type=str,
                        default=os.environ.get("VLM_API_KEY", ""),
                        help="API key for the server (if required). "
                             "Default: env VLM_API_KEY.")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help=f"Max tokens to generate (default: {MAX_TOKENS}).")
    parser.add_argument("--max-image-side", type=int, default=MAX_IMAGE_SIDE,
                        help=f"Downscale image longest edge to at most this "
                             f"(default: {MAX_IMAGE_SIDE}; use 1280 for Qwen3-VL).")
    parser.add_argument("--test-cases", type=str, default=TEST_CASES_FILE,
                        help=f"Test cases JSON file (default: {TEST_CASES_FILE}).")
    parser.add_argument("--api-key-judge", type=str, default=None,
                        help="OpenRouter API key for judge scoring (overrides env).")
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL,
                        help=f"Model to use for judge scoring (default: {JUDGE_MODEL}).")
    args = parser.parse_args()

    if args.api_key_judge:
        OPENROUTER_API_KEY = args.api_key_judge
    JUDGE_MODEL = args.judge_model

    if not args.base_url:
        print("ERROR: --base-url is required (or set VLM_API_BASE_URL in .env).")
        sys.exit(1)
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set — needed for judge scoring.")
        sys.exit(1)

    # Resolve model id
    if args.model:
        model_name = args.model
    else:
        print(f"No --model given; querying {args.base_url}/v1/models ...")
        model_name = discover_model(args.base_url, args.api_key)
        if not model_name:
            print("ERROR: could not determine model id. Pass --model explicitly.")
            sys.exit(1)
        print(f"  Using model: {model_name}")

    # Load test cases
    with open(args.test_cases) as f:
        tests = json.load(f)

    # Validate images
    for i, case in enumerate(tests, 1):
        img = case.get("image", "")
        if not img:
            print(f"ERROR: Test case {i} has no image specified")
            sys.exit(1)
        if not os.path.exists(img):
            print(f"ERROR: Image not found: {img} (test case {i})")
            sys.exit(1)

    print("=" * 70)
    print("  CUSTOM-API VLM BENCHMARK SUITE")
    print(f"  Endpoint:  {args.base_url}")
    print(f"  Model:     {model_name}")
    print(f"  Test cases: {len(tests)}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Max image side: {args.max_image_side}px")
    print(f"  Judge:      {JUDGE_MODEL} (via OpenRouter)")
    print("=" * 70)

    summary = run_benchmark(
        model_name, args.base_url, args.api_key, tests,
        args.max_tokens, args.max_image_side,
    )

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    model_label = sanitize_model_name(model_name)

    json_output = os.path.join(output_dir, f"api_results_{model_label}_{timestamp}.json")
    with open(json_output, "w") as f:
        json.dump([summary], f, indent=2)
    print(f"\nCombined results saved to: {json_output}")

    html_output = os.path.join(output_dir, f"api_results_{model_label}_{timestamp}.html")
    generate_html_report([summary], html_output)
    print(f"HTML report saved to: {html_output}")

    print("\n" + "=" * 70)
    print("  Custom-API VLM Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
