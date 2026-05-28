# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

This is a VLM (Vision-Language Model) benchmark suite that evaluates how well models answer questions about PC screenshots. It supports two execution modes:

- **Local inference** (`local_benchmark.py`): runs quantized VLMs on Apple Silicon via `mlx_vlm`
- **API inference** (`openrouter_benchmark.py`): sends images to cloud models via OpenRouter

Both scripts use an **LLM-as-judge** pattern: after each model answers a question, `openai/gpt-5.4` (via OpenRouter) scores the response 0–10 against a grounded answer. Results are saved to `output/` as JSON and HTML.

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

### Response parsing (local only)

Model output is split on `</think>` to strip chain-of-thought reasoning before scoring: `result.text.split('</think>')[-1].strip()`
