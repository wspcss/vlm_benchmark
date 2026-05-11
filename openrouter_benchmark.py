#!/usr/bin/env python3
"""
OpenRouter VLM Benchmark Script
Benchmarks vision-language models via the OpenRouter API with single-image support.
Supports benchmarking a single model or all models from openrouter_models.json.
"""

import argparse
import json
import time
import os
import sys
import base64
import re

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
DEFAULT_MODEL = "openai/gpt-4.1-mini"
# JUDGE_MODEL = "openai/gpt-5"
JUDGE_MODEL = "openai/gpt-5.4-mini"
OPENROUTER_MODELS_FILE = "openrouter_models.json"
TEST_CASES_FILE = "test_case.json"
MAX_TOKENS = 512


def sanitize_model_name(model_name):
    return model_name.replace("/", "_").strip()


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
    """Call the OpenRouter chat completions API."""
    import urllib.request
    import urllib.error

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
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
        f"on a scale of 0 to 10, where 0 is completely wrong and 10 is a perfect match. "
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


def run_benchmark(model, tests, max_tokens):
    """Run benchmark for a single model via OpenRouter API."""
    safe_name = sanitize_model_name(model)

    print(f"\n{'=' * 70}")
    print(f"OPENROUTER VLM BENCHMARK: {model}")
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

        print(f"\n  Image: {image_path}")
        image_b64 = encode_image_to_base64(image_path)

        # Build messages
        content = [
            {"type": "image_url", "image_url": {"url": image_b64}},
            {"type": "text", "text": question},
        ]
        messages = [{"role": "user", "content": content}]

        try:
            response_data, elapsed = call_openrouter(model, messages, max_tokens)
        except Exception as e:
            print(f"\n  ❌ API call failed: {e}")
            results.append({
                "question": question,
                "model_response": f"ERROR: {e}",
                "grounded_answer": grounded,
                "accuracy_score": None,
                "time_seconds": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "tokens_per_second": 0,
            })
            continue

        # Parse response
        choices = response_data.get("choices", [])
        if choices:
            response_text = choices[0].get("message", {}).get("content", "").strip()
        else:
            response_text = ""

        usage = response_data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        tps = output_tokens / elapsed if elapsed > 0 else 0

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
        print(f"  Time:   {elapsed:.2f}s | Input: {input_tokens} | Output: {output_tokens} | TPS: {tps:.1f}")

        results.append({
            "question": question,
            "model_response": response_text,
            "grounded_answer": grounded,
            "accuracy_score": accuracy_score,
            "time_seconds": round(elapsed, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_second": round(tps, 1),
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
    print(f"Model:            {model}")
    print(f"Image:            {image_path}")
    print(f"Test cases:       {len(results)}")
    if avg_score is not None:
        print(f"Avg judge score:  {avg_score:.1f}/10 ({total_scored} scored)")
    print(f"Total time:       {total_elapsed:.2f}s")
    print(f"Avg per case:     {avg_time:.2f}s")
    print(f"Avg TPS:          {avg_tps:.1f}")
    print(f"Total tokens in:  {total_input_tokens}")
    print(f"Total tokens out: {total_output_tokens}")

    summary = {
        "model": model,
        "image": image_path,
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


def generate_html_report(summaries, output_path):
    """Generate an HTML report from benchmark results."""
    import html

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Build model comparison rows
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
          <td>{html.escape(s['model'])}</td>
          <td>{s['avg_tokens_per_second']:.1f}</td>
          <td>{s['avg_per_case_s']:.2f}s</td>
          <td>{score_badge}</td>
          <td>{s['total_input_tokens']}</td>
          <td>{s['total_output_tokens']}</td>
        </tr>"""

    # Build per-model detail sections
    model_sections = ""
    for s in summaries:
        case_rows = ""
        for i, r in enumerate(s.get("results", []), 1):
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
            case_rows += f"""
            <tr>
              <td>{i}</td>
              <td>{html.escape(r.get('question', ''))}</td>
              <td style="max-width:600px;">{response_escaped}</td>
              <td>{html.escape(r.get('grounded_answer', ''))}</td>
              {score_cell}
              <td>{r.get('time_seconds', 0):.2f}s</td>
              <td>{r.get('tokens_per_second', 0):.1f}</td>
            </tr>"""

        avg_judge = s.get("avg_judge_score")
        if avg_judge is not None:
            if avg_judge >= 7:
                summary_score = f'<span style="color:#16a34a;font-weight:bold;font-size:1.2em;">{avg_judge:.1f}/10</span>'
            elif avg_judge >= 4:
                summary_score = f'<span style="color:#d97706;font-weight:bold;font-size:1.2em;">{avg_judge:.1f}/10</span>'
            else:
                summary_score = f'<span style="color:#dc2626;font-weight:bold;font-size:1.2em;">{avg_judge:.1f}/10</span>'
        else:
            summary_score = '<span style="color:#999;">N/A</span>'

        model_sections += f"""
    <div class="model-section">
      <h2>{html.escape(s['model'])}</h2>
      <div class="summary-grid">
        <div class="stat-card">
          <div class="stat-label">Avg Judge Score</div>
          <div class="stat-value">{summary_score}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Avg TPS</div>
          <div class="stat-value">{s['avg_tokens_per_second']:.1f}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Avg Time/Case</div>
          <div class="stat-value">{s['avg_per_case_s']:.2f}s</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Total Tokens</div>
          <div class="stat-value">{s['total_input_tokens']} / {s['total_output_tokens']}</div>
        </div>
      </div>
      <table>
        <thead>
          <tr><th>#</th><th>Question</th><th>Model Response</th><th>Expected</th><th>Score</th><th>Time</th><th>TPS</th></tr>
        </thead>
        <tbody>{case_rows}
        </tbody>
      </table>
    </div>"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VLM Benchmark Results - {timestamp}</title>
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
    .model-section {{ background: white; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .model-section h2 {{ font-size: 1.2rem; margin-bottom: 1rem; color: #334155; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }}
    .stat-card {{ background: #f8fafc; border-radius: 8px; padding: 1rem; text-align: center; }}
    .stat-label {{ font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem; }}
    .stat-value {{ font-size: 1.1rem; font-weight: 600; color: #0f172a; }}
    .badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>🔬 VLM Benchmark Results</h1>
    <p class="timestamp">Generated: {timestamp} | Models: {len(summaries)} | Judge: {html.escape(JUDGE_MODEL)}</p>

    <div class="comparison">
      <h2>Model Comparison</h2>
      <table>
        <thead>
          <tr><th>Model</th><th>Avg TPS</th><th>Avg/Case</th><th>Avg Score</th><th>Tokens In</th><th>Tokens Out</th></tr>
        </thead>
        <tbody>{comparison_rows}
        </tbody>
      </table>
    </div>

    {model_sections}
  </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def main():
    parser = argparse.ArgumentParser(description="OpenRouter VLM Benchmark")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Single model to benchmark (e.g., {DEFAULT_MODEL})")
    parser.add_argument("--all", action="store_true",
                        help="Benchmark all models in openrouter_models.json")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help=f"Max tokens to generate (default: {MAX_TOKENS})")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenRouter API key (overrides built-in key)")
    args = parser.parse_args()

    # Override API key if provided
    global API_KEY
    if args.api_key:
        API_KEY = args.api_key

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
        with open(OPENROUTER_MODELS_FILE) as f:
            models_config = json.load(f)
        models = [entry["name"] if isinstance(entry, dict) else entry for entry in models_config]
        print(f"Found {len(models)} model(s) in {OPENROUTER_MODELS_FILE}")
    elif args.model:
        models = [args.model]
    else:
        print("ERROR: Specify --model <model_id> or --all")
        sys.exit(1)

    print("=" * 70)
    print("  OPENROUTER VLM BENCHMARK SUITE")
    print(f"  Models:     {len(models)}")
    print(f"  Test cases: {len(tests)}")
    print(f"  Max tokens: {args.max_tokens}")
    print("=" * 70)

    summaries = []
    failed = []

    for model in models:
        try:
            summary = run_benchmark(model, tests, args.max_tokens)
            summaries.append(summary)
        except Exception as e:
            print(f"\n  ❌ FAILED: {model}: {e}")
            failed.append(model)
            continue

    if failed:
        print(f"\n⚠️  {len(failed)} model(s) failed:")
        for f_name in failed:
            print(f"   - {f_name}")

    # Comparison table
    if len(summaries) > 1:
        print("\n\n" + "=" * 70)
        print("  OPENROUTER MODEL COMPARISON")
        print("=" * 70)

        print(f"\n{'Model':<40} {'Avg TPS':>10} {'Avg/Case':>10} {'Avg Score':>10}")
        print("-" * 72)
        for s in summaries:
            short = s["model"]
            avg_judge = s.get("avg_judge_score")
            score_str = f"{avg_judge}/10" if avg_judge is not None else "N/A"
            print(f"{short:<40} {s['avg_tokens_per_second']:>9.1f} {s['avg_per_case_s']:>9.2f}s {score_str:>10}")

    # Save combined results to a single JSON file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    combined_output = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(combined_output, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nCombined results saved to: {combined_output}")

    # Generate HTML report
    html_output = os.path.join(output_dir, f"results_{timestamp}.html")
    generate_html_report(summaries, html_output)
    print(f"HTML report saved to: {html_output}")

    print("\n" + "=" * 70)
    print("  OpenRouter Benchmarks complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
