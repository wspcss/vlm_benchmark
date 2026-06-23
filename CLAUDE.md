# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

This is a VLM (Vision-Language Model) benchmark suite that evaluates how well models answer questions about PC screenshots. It supports two execution modes:

- **Local inference** (`local_benchmark.py`): runs quantized VLMs on Apple Silicon via `mlx_vlm`
- **API inference** (`openrouter_benchmark.py`): sends images to cloud models via OpenRouter

Both scripts use an **LLM-as-judge** pattern: after each model answers a question, `openai/gpt-5.4` (via OpenRouter) scores the response 0–10 against a grounded answer. Results are saved to `output/` as JSON and HTML.

## Image-size stability findings (read `progress.md` for full detail)

`benchmark_image_sizes.py` tested every local model across an image-resolution ladder
(360→3840px, `images/size/`) to find the **max image size each model handles before it
crashes/hangs the Metal GPU or emits degenerate output**. Full detail, methodology, and the
GPU-contamination fixes are in **`progress.md`**; the data is in
`output/size_matrix_*.{json,html}` (sorted by provider/family/size). New sessions: start from
`progress.md` for this topic.

**Max supported resolution by family** (longest edge; failure is per-family, size-monotonic,
and set by the image processor's prefill token count):

| Max | Families |
|---|---|
| **3840px (4K, no limit)** | Gemma-3, Gemma-4, Llama-3.2-Vision, GLM-4.5V/4.6V, Qwen2-VL, Qwen2.5-VL (7B/32B), Qwen3.5, Qwen3.6, deepseek-vl2 (small/4bit) |
| **1280px** | **Qwen3-VL (2B→32B)** — fails at 1920px ≈ 2061 prefill tokens; the only family with a real low ceiling |
| **960px** | Qwen2.5-VL-3B |
| **none** | deepseek-vl2-tiny (degenerate), llava-1.5 / llava-v1.6 (compat error) |

`local_benchmark.py`'s `MAX_IMAGE_SIDE` exists to downscale screenshots under these limits.

## Running benchmarks

Requires `OPENROUTER_API_KEY` in `.env` (sourced automatically by the shell scripts).

```bash
# Run all local models defined in LOCAL_MODELS_FILE (currently local_kimi_models.json)
./run_local_benchmark.sh

# Run all OpenRouter API models in models.json
./run_benchmark.sh

# Run a single local model
python3.11 local_benchmark.py --model mlx-community/Qwen3-VL-4B-Instruct-4bit

# Run a single API model
python openrouter_benchmark.py --model openai/gpt-4o-mini

# Use a different test cases file or increase output tokens
python3.11 local_benchmark.py --model <id> --max-tokens 1024
```

**Important:** `LOCAL_MODELS_FILE` in `local_benchmark.py:28` is hardcoded to `local_kimi_models.json`. To use `--all` with a different model family, edit that constant before running.

## Architecture

### Data flow

```
test_case.json  ──►  run inference (mlx_vlm or OpenRouter API)
                         │
                         ▼
                  judge_response()  ◄──  OpenRouter (gpt-5.4)
                         │
                         ▼
              output/local_results_<label>_<timestamp>.{json,html}
```

### Model config files

Each `*_models.json` file is a list of `{"name": "<model_id>"}` objects (or a flat list of strings). Local model IDs are `mlx-community/` HuggingFace paths; OpenRouter IDs follow `provider/model` format.

`models.json` (OpenRouter) also supports a top-level `"provider"` key to pin routing:
```json
{ "provider": {"order": ["openai"]}, "models": [...] }
```

### Test cases

`test_case.json` — full multi-image benchmark suite  
`test_case_single.json` — single test case for quick iteration

Each entry:
```json
{
  "image": "images/win_multi_window.png",
  "question": "...",
  "grounded_answer": "...",
  "category": "Application enumeration"
}
```

### Key constants (local_benchmark.py)

| Constant | Default | Purpose |
|---|---|---|
| `LOCAL_MODELS_FILE` | `local_kimi_models.json` | Model list used by `--all` |
| `TEST_CASES_FILE` | `test_case.json` | Test cases to run |
| `JUDGE_MODEL` | `openai/gpt-5.4` | Scoring model via OpenRouter |
| `MAX_TOKENS` | `512` | Generation limit per response |

### Memory management (local only)

Models are explicitly deleted and `gc.collect()` called between each model load to free Apple Silicon unified memory before loading the next model.

### Suppressing chain-of-thought (local only)

`run_local_inference` passes `enable_thinking=False` to `apply_chat_template`. `mlx_vlm` forwards this kwarg down to the model's HuggingFace chat template, where thinking models honor it by pre-filling a closed `<think></think>` block so **no reasoning is generated**:

- **Qwen3.5 / Qwen3.6** — template emits `<think>\n\n</think>\n\n` at the generation prompt.
- **GLM-4.5V / GLM-4.6V** — template appends `/nothink` to the user turn and prepends `<think></think>`.

Non-thinking templates (Qwen2-VL, Gemma, Llama-Vision, deepseek-vl2) accept the kwarg as an unused variable and ignore it, so it's safe to pass unconditionally. Suppressing at the source avoids spending generation tokens on reasoning.

### Response parsing (local only)

`enable_thinking=False` handles the reasoning, so no chain-of-thought stripping is needed. The one remaining cleanup is GLM-4.6V's final-answer box markers: it wraps its answer in the special tokens `<|begin_of_box|>…<|end_of_box|>`. These are emitted by the **model during generation** (not by the chat template), so there is no prompt-side flag to suppress them — they must be stripped after the fact.

`strip_box_markers()` (in `local_benchmark.py`) prefers the boxed content if present, otherwise removes any stray marker. Applied in `run_local_inference` so both the saved `model_response` and the judge input are clean; it's a no-op for all other models.
