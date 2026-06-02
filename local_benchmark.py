#!/usr/bin/env python3
"""
Local VLM Benchmark Script
Benchmarks vision-language models running locally via mlx_vlm.
Uses OpenRouter API for LLM judge scoring.
Supports benchmarking a single model or all models from localmodels.json.
"""

import argparse
import gc
import json
import os
import re
import sys
import time
import base64

# Local inference imports
from PIL import Image
from mlx_vlm import load as mlx_load
from mlx_vlm.generate import generate as mlx_generate
from mlx_vlm.prompt_utils import apply_chat_template

# OpenRouter judge constants
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
# JUDGE_MODEL = "openai/gpt-5.4"
JUDGE_MODEL = "deepseek/deepseek-v4-flash"

LOCAL_MODELS_FILES = ["local_qwen_models.json"
                    , "local_gemini_models.json"
                    , "local_glm_models.json"
                    , "local_meta_models.json"
                    , "local_deepseek_models.json"
]

TEST_CASES_FILE = "test_case.json"
MAX_TOKENS = 512

def provider_key_from_file(models_file):
    """Derive a provider key from a *_models.json filename.

    local_qwen_models.json -> "qwen". Fully filename-driven, so adding a new
    local_<provider>_models.json to LOCAL_MODELS_FILES groups it automatically.
    """
    base = os.path.splitext(os.path.basename(models_file))[0]
    return base.replace("local_", "", 1).replace("_models", "")


def provider_display(key):
    """Human-readable label for a provider key (e.g. 'qwen' -> 'Qwen')."""
    return key.capitalize() if key else "Other"


def provider_for_model(model_name):
    """Best-effort provider for a single model run (no source file known).

    Matches the model id against the provider keys discovered from
    LOCAL_MODELS_FILES, so it stays in sync with whatever files are configured.
    """
    n = model_name.lower()
    for models_file in LOCAL_MODELS_FILES:
        key = provider_key_from_file(models_file)
        if key and key in n:
            return provider_display(key)
    return "Other"


# Hard cap on an image's longest edge before inference. Large screenshots
# (e.g. 2880x1800) otherwise expand to 5000-7000+ prefill tokens, which triggers
# Metal GPU faults / degenerate output. Aspect ratio is preserved (resize, not crop).
# 1920 keeps native 1920x1080 untouched (~2600 prefill tokens, safely under the
# fault threshold) while still downscaling larger 4K-ish screenshots.
MAX_IMAGE_SIDE = 1920


def sanitize_model_name(model_name):
    return model_name.replace("/", "_").strip()


def load_image(image_path, max_side=MAX_IMAGE_SIDE):
    """Open an image and downscale so its longest edge is <= max_side."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return img


def encode_image_to_base64(image_path):
    """Read an image file and return a base64 data URI."""
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


def call_openrouter(model, messages, max_tokens):
    """Call the OpenRouter chat completions API (used for judge scoring)."""
    import urllib.request
    import urllib.error

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        OPENROUTER_API_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            response_data = json.loads(body)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"API error {e.code}: {error_body}")
    elapsed = time.time() - start

    return response_data, elapsed


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
            # Check content first, then fall back to reasoning field
            text = (msg.get("content") or "").strip()
            if not text:
                reasoning = (msg.get("reasoning") or "").strip()
                if reasoning:
                    text = reasoning
            # Extract the first number (0-10) found in the response
            num_match = re.search(r'\b(10|\d)\b', text)
            if num_match:
                return int(num_match.group())
        return None
    except Exception as e:
        print(f"    ⚠️  Judge API call failed: {e}")
        return None


def run_local_inference(model, processor, image_path, question, max_tokens):
    """Run inference on a local mlx_vlm model and return results."""
    # Build the prompt using the model's chat template
    prompt = apply_chat_template(
        processor,
        model.config,
        question,
        num_images=1,
    )

    image = load_image(image_path)

    start = time.time()
    result = mlx_generate(
        model,
        processor,
        prompt=prompt,
        image=image,
        max_tokens=max_tokens,
        temperature=0.0,
        verbose=False,
    )
    elapsed = time.time() - start

    return {
        "response_text": result.text.strip(),
        "prompt_tokens": result.prompt_tokens,
        "generation_tokens": result.generation_tokens,
        "generation_tps": result.generation_tps,
        "peak_memory_gb": result.peak_memory,
        "time_seconds": elapsed,
    }


def run_benchmark(model_name, model, processor, tests, max_tokens):
    """Run benchmark for a single local model."""
    print(f"\n{'=' * 70}")
    print(f"LOCAL VLM BENCHMARK: {model_name}")
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
            inference = run_local_inference(
                model, processor, image_path, question, max_tokens
            )
        except Exception as e:
            print(f"\n  ❌ Local inference failed: {e}")
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
                "peak_memory_gb": 0,
            })
            continue

        response_text = inference["response_text"]
        input_tokens = inference["prompt_tokens"]
        output_tokens = inference["generation_tokens"]
        tps = inference["generation_tps"]
        elapsed = inference["time_seconds"]
        peak_memory = inference["peak_memory_gb"]

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        # Check accuracy using LLM judge
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
        print(f"  Time:   {elapsed:.2f}s | Input: {input_tokens} | Output: {output_tokens} | TPS: {tps:.1f} | Peak Mem: {peak_memory:.2f} GB")

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
            "peak_memory_gb": round(peak_memory, 2),
        })

    total_elapsed = time.time() - total_start

    # Compute summary stats
    valid_results = [r for r in results if r["time_seconds"] > 0]
    avg_tps = (
        sum(r["tokens_per_second"] for r in valid_results) / len(valid_results)
        if valid_results
        else 0
    )
    avg_time = (
        sum(r["time_seconds"] for r in valid_results) / len(valid_results)
        if valid_results
        else 0
    )
    avg_peak_memory = (
        sum(r["peak_memory_gb"] for r in valid_results) / len(valid_results)
        if valid_results
        else 0
    )

    scored_results = [r for r in results if r["accuracy_score"] is not None]
    total_scored = len(scored_results)
    avg_score = (
        sum(r["accuracy_score"] for r in scored_results) / total_scored
        if total_scored > 0
        else None
    )

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Model:            {model_name}")
    print(f"Test cases:       {len(results)}")
    if avg_score is not None:
        print(f"Avg judge score:  {avg_score:.1f}/10 ({total_scored} scored)")
    print(f"Total time:       {total_elapsed:.2f}s")
    print(f"Avg per case:     {avg_time:.2f}s")
    print(f"Avg TPS:          {avg_tps:.1f}")
    print(f"Avg peak memory:  {avg_peak_memory:.2f} GB")
    print(f"Total tokens in:  {total_input_tokens}")
    print(f"Total tokens out: {total_output_tokens}")

    summary = {
        "model": model_name,
        "total_cases": len(results),
        "scored_cases": total_scored,
        "avg_judge_score": round(avg_score, 1) if avg_score is not None else None,
        "total_time_s": round(total_elapsed, 2),
        "avg_per_case_s": round(avg_time, 2),
        "avg_tokens_per_second": round(avg_tps, 1),
        "avg_peak_memory_gb": round(avg_peak_memory, 2),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "results": results,
    }

    return summary


def generate_html_report(summaries, output_path):
    """Generate an HTML report from benchmark results."""
    import html

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Group summaries by provider for the comparison table, preserving the order
    # in which providers first appear.
    grouped = {}
    for s in summaries:
        grouped.setdefault(s.get("provider") or "Other", []).append(s)
    show_provider_headers = len(grouped) > 1

    # Build model comparison rows, with a header row per provider group
    comparison_rows = ""
    for provider, group in grouped.items():
        if show_provider_headers:
            comparison_rows += f"""
        <tr class="provider-row">
          <td colspan="7">{html.escape(provider)}</td>
        </tr>"""
        for s in group:
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
          <td>{html.escape(s['model'])}</td>
          <td>{s['avg_tokens_per_second']:.1f}</td>
          <td>{s['avg_per_case_s']:.2f}s</td>
          <td>{score_badge}</td>
          <td>{s.get('avg_peak_memory_gb', 0):.2f} GB</td>
          <td>{s['total_input_tokens']}</td>
          <td>{s['total_output_tokens']}</td>
        </tr>"""

    # Build per-test-case sections (image/question/expected shown once, model responses compared)
    max_cases = max((len(s.get("results", [])) for s in summaries), default=0)
    test_case_sections = ""
    for case_idx in range(max_cases):
        # Get shared test case info from the first model that has this case
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

        # Embed image once
        image_html = ""
        if img_path and os.path.exists(img_path):
            img_b64 = encode_image_to_base64(img_path)
            image_html = f'<img src="{img_b64}" alt="{html.escape(img_path)}" style="max-height:300px;border-radius:8px;border:1px solid #e2e8f0;" />'

        # Build comparison rows: one row per model
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

            response_escaped = html.escape(r.get("model_response", ""))
            comparison_rows_html += f"""
            <tr>
              <td>{html.escape(s['model'])}</td>
              <td style="max-width:600px;">{response_escaped}</td>
              {score_cell}
              <td>{r.get('time_seconds', 0):.2f}s</td>
              <td>{r.get('tokens_per_second', 0):.1f}</td>
              <td>{r.get('peak_memory_gb', 0):.2f} GB</td>
            </tr>"""

        category_badge = f'<span style="background:#e0e7ff;color:#3730a3;padding:0.2rem 0.6rem;border-radius:9999px;font-size:0.8rem;font-weight:600;">{html.escape(category)}</span>' if category else ''
        test_case_sections += f"""
    <div class="test-case-section">
      <h2>Test Case {case_idx + 1} {category_badge}</h2>
      <div class="case-info">
        {image_html}
        <div class="case-details">
          <p><strong>Question:</strong> {html.escape(question)}</p>
          <p><strong>Expected Answer:</strong> {html.escape(grounded)}</p>
        </div>
      </div>
      <table>
        <thead>
          <tr><th>Model</th><th>Response</th><th>Score</th><th>Time</th><th>TPS</th><th>Peak Mem</th></tr>
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
  <title>Local VLM Benchmark Results - {timestamp}</title>
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
    .provider-row td {{ background: #eef2ff; color: #3730a3; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 2px solid #c7d2fe; }}
    .provider-row:hover td {{ background: #eef2ff; }}
    .test-case-section {{ background: white; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .test-case-section h2 {{ font-size: 1.2rem; margin-bottom: 1rem; color: #334155; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; }}
    .case-info {{ display: flex; gap: 1.5rem; margin-bottom: 1.5rem; align-items: flex-start; flex-wrap: wrap; }}
    .case-details {{ flex: 1; min-width: 300px; }}
    .case-details p {{ margin-bottom: 0.5rem; line-height: 1.5; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>🖥️ Local VLM Benchmark Results</h1>
    <p class="timestamp">Generated: {timestamp} | Models: {len(summaries)} | Judge: {html.escape(JUDGE_MODEL)}</p>

    <div class="comparison">
      <h2>Model Comparison</h2>
      <table>
        <thead>
          <tr><th>Model</th><th>Avg TPS</th><th>Avg/Case</th><th>Avg Score</th><th>Avg Peak Mem</th><th>Tokens In</th><th>Tokens Out</th></tr>
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


def main():
    global API_KEY, JUDGE_MODEL

    parser = argparse.ArgumentParser(description="Local VLM Benchmark (mlx_vlm)")
    parser.add_argument("--model", type=str, default=None,
                        help="Single model to benchmark (e.g., mlx-community/Qwen3-VL-4B-Instruct-4bit)")
    parser.add_argument("--all", action="store_true",
                        help="Benchmark all models in localmodels.json")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help=f"Max tokens to generate (default: {MAX_TOKENS})")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenRouter API key for judge scoring (overrides env var)")
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL,
                        help=f"Model to use for judge scoring (default: {JUDGE_MODEL})")
    args = parser.parse_args()

    # Override API key if provided
    if args.api_key:
        API_KEY = args.api_key
    JUDGE_MODEL = args.judge_model

    # Load test cases
    with open(TEST_CASES_FILE) as f:
        tests = json.load(f)

    # Validate images in test cases
    for i, case in enumerate(tests, 1):
        img = case.get("image", "")
        if not img:
            print(f"ERROR: Test case {i} has no image specified")
            sys.exit(1)
        if not os.path.exists(img):
            print(f"ERROR: Image not found: {img} (test case {i})")
            sys.exit(1)

    # Determine which models to benchmark
    if args.all:
        models = []
        for models_file in LOCAL_MODELS_FILES:
            with open(models_file) as f:
                models_config = json.load(f)
            provider = provider_display(provider_key_from_file(models_file))
            models += [
                (entry["name"] if isinstance(entry, dict) else entry, provider)
                for entry in models_config
            ]
        print(f"Found {len(models)} model(s) across {LOCAL_MODELS_FILES}")
    elif args.model:
        models = [(args.model, provider_for_model(args.model))]
    else:
        print("ERROR: Specify --model <model_id> or --all")
        sys.exit(1)

    print("=" * 70)
    print("  LOCAL VLM BENCHMARK SUITE (mlx_vlm)")
    print(f"  Models:     {len(models)}")
    print(f"  Test cases: {len(tests)}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Judge:      {JUDGE_MODEL} (via OpenRouter)")
    print("=" * 70)

    summaries = []
    failed = []

    for model_name, provider in models:
        print(f"\n{'─' * 70}")
        print(f"  Loading model: {model_name} ({provider})")
        print(f"{'─' * 70}")

        try:
            model, processor = mlx_load(model_name)
        except Exception as e:
            print(f"\n  ❌ Failed to load model: {e}")
            failed.append(model_name)
            continue

        try:
            summary = run_benchmark(model_name, model, processor, tests, args.max_tokens)
            summary["provider"] = provider
            summaries.append(summary)
        except Exception as e:
            print(f"\n  ❌ FAILED: {model_name}: {e}")
            failed.append(model_name)
        finally:
            # Unload model to free memory before loading the next one
            del model
            del processor
            gc.collect()

    if failed:
        print(f"\n⚠️  {len(failed)} model(s) failed:")
        for f_name in failed:
            print(f"   - {f_name}")

    # Comparison table
    if len(summaries) > 1:
        print("\n\n" + "=" * 70)
        print("  LOCAL MODEL COMPARISON")
        print("=" * 70)

        print(f"\n{'Model':<50} {'Avg TPS':>10} {'Avg/Case':>10} {'Avg Score':>10} {'Peak Mem':>10}")
        print("-" * 92)
        for s in summaries:
            short = s["model"]
            avg_judge = s.get("avg_judge_score")
            score_str = f"{avg_judge}/10" if avg_judge is not None else "N/A"
            print(f"{short:<50} {s['avg_tokens_per_second']:>9.1f} {s['avg_per_case_s']:>9.2f}s {score_str:>10} {s.get('avg_peak_memory_gb', 0):>9.2f} GB")

    # Build a model label for filenames
    if args.all:
        model_label = "+".join(os.path.splitext(f)[0] for f in LOCAL_MODELS_FILES)
    elif args.model:
        model_label = sanitize_model_name(args.model)
    else:
        model_label = "unknown"

    # Save combined results to a single JSON file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    combined_output = os.path.join(output_dir, f"local_results_{model_label}_{timestamp}.json")
    with open(combined_output, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nCombined results saved to: {combined_output}")

    # Generate HTML report
    html_output = os.path.join(output_dir, f"local_results_{model_label}_{timestamp}.html")
    generate_html_report(summaries, html_output)
    print(f"HTML report saved to: {html_output}")

    print("\n" + "=" * 70)
    print("  Local VLM Benchmarks complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()