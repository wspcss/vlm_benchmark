# Progress — Image-size stability matrix for local VLMs

_Last updated: 2026-06-18. GPU-contamination fix DONE; full clean matrix produced._

## STATUS: COMPLETE ✅

The GPU-contamination cascade is fixed (options 1+2+4 implemented) and a fully clean
33-model matrix is in **`output/size_matrix_20260618_112227.{json,html}`**.

### Final findings (max supported resolution)
- **3840px (no limit):** Gemma-3/4 (all), Llama-3.2-Vision 4/8bit, GLM-4.5V/4.6V/Flash,
  Qwen2-VL 2B/7B, Qwen2.5-VL 7B/32B, Qwen3.5 2B/4B/9B/27B/35B, Qwen3.6 27B/35B,
  deepseek-vl2 small/4bit.
- **1280px:** Qwen3-VL 2B/4B/8B/30B-A3B/32B (fail at 1920px ≈ 2061 prefill tokens). The only
  family with a real low ceiling; consistent across 2B→32B.
- **960px:** Qwen2.5-VL-3B (garbage at 1280px).
- **none:** deepseek-vl2-tiny (garbage at 360px), llava-1.5/v1.6 (compat error, won't run).

### Fix implemented (in benchmark_image_sizes.py)
- **(1) Graceful kill + scaled timeout:** `terminate()` does SIGTERM + 12s grace before
  SIGKILL (abrupt SIGKILL mid-GPU-op can itself wedge Metal); `timeout_for()` gives ≥20B
  models 600s (vs 240s) so merely-slow models aren't false-killed.
- **(2) GPU canary + recovery + retry:** `gpu_healthy()` runs a model-free Metal matmul in a
  throwaway process; `wait_for_gpu()` polls until recovery. After any crash/hang the next
  model waits for a healthy GPU; a crash/hang at prefill < 1500 (`suspect_contamination`) is
  treated as leftover contamination and the model is re-run once on a verified-healthy GPU.
- **(4) Smallest-params-first ordering** (`param_billions`) so hang-prone 32B/35B run last and
  can't poison the rest.
- Parent prints now `flush=True` so the run is monitorable live.
- One transient remained (Qwen3.5-35B-A3B hung at 960px even after retry, at run position #32);
  an isolated re-run confirmed it actually handles 3840px, and that row was patched into the
  authoritative report.

### Per-failure confirmation (`--confirm`)
Added an **isolation re-run** mode to prove each failure is real and not residual contamination:
```bash
python3.11 benchmark_image_sizes.py --confirm [matrix.json]   # default: latest matrix
python3.11 benchmark_image_sizes.py --models <id> --sizes 1920 # run one (model,size) alone
```
- New `--sizes` filter (honored by `discover_sizes` in both parent and worker via `SIZE_FILTER`,
  passed to the worker subprocess as `--sizes`).
- `confirm_failures()` reads a matrix JSON, finds each model's first observed failure
  (crash/hang/degenerate/error), and re-runs **only that size in a fresh process** (nothing runs
  before it). Still fails => real limit; now passes => was contamination. Saves
  `output/size_matrix_*_confirm.json`.
- **Result (run on a fresh PC reboot — cleanest baseline): 9/9 failures CONFIRMED real,
  0 contamination.** Saved `output/size_matrix_20260618_112227_confirm.json`.
- **Important nuance:** the failure *mode* shifted on a clean GPU — cases that `crash`/`hang`ed
  in the big sequential run came up `degenerate` (garbage output, no hard abort) in isolation:
  Qwen3-VL-4B/8B/30B-A3B@1920 (crash->degenerate), Qwen3-VL-32B@1920 (hang->degenerate). So the
  pass/fail BOUNDARY is deterministic (1920px fails, 1280px passes for all Qwen3-VL), but whether
  it manifests as garbage vs. a hard GPU crash depends on accumulated GPU/memory pressure.
  deepseek-vl2-tiny@360, Qwen3-VL-2B@1920, Qwen2.5-VL-3B@1280 stayed degenerate; llava@360 stayed
  a (deterministic) compat error.

### New knowledge / lessons
- **Clean SIGABRT crashes do NOT wedge the GPU** (next process runs fine); only **GPU
  timeouts/hangs** (`kIOGPUCommandBufferCallbackErrorTimeout/Hang`) do — often made worse by our
  own SIGKILL mid-GPU-op. That distinction drove the fix design.
- **Failure is per-family and size-monotonic**, and the crash *threshold* is consistent across
  model sizes within a family (Qwen3-VL 2B→32B all fail at 1920px ≈ 2061 prefill).
- **Prefill scales superlinearly with longest edge** (Qwen3-VL: 360→109, 1280→901, 1920→2061,
  3840→8183 tokens) — but is **flat** for Gemma-3 (281) and Llama-3.2 (22), which are size-immune.
- **Qwen2/2.5-VL tolerate huge prefills** (10k+ tokens at 3840px) and stay coherent — they do NOT
  have the Qwen3-VL ceiling.
- Big MoE models (35B-A3B) can throw a **transient** hang at a small size under memory pressure;
  an isolated re-run is the tiebreaker (Qwen3.5-35B-A3B: 640px transient -> really 3840px).
- **WindowServer** (macOS compositor, user `_windowserver`) is normal; a few % CPU at idle is
  fine. Sustained high WindowServer CPU after heavy Metal work can be a driver hiccup — log
  out/in to restart it cleanly. After the run, no benchmark processes persist; system returns idle.
- macOS pops a "Python quit unexpectedly" dialog per worker crash (expected). Suppressed with
  `defaults write com.apple.CrashReporter DialogType none` (revert: `... DialogType crashreport`).
  Each crash also writes a `~/Library/Logs/DiagnosticReports/Python-*.ips` report (safe to delete).

---
## (Original notes below — kept for reference)

## What this is

A new script, **`benchmark_image_sizes.py`**, that runs each local VLM against a ladder of
screenshot resolutions (`images/size/windows-{360,480,640,960,1280,1920,2560,3840}.jpg`) and
reports, per model, which image sizes are **supported / degenerate / crash / error**, plus each
model's **max supported resolution**. Output: `output/size_matrix_<ts>.{json,html}`.

## Why it exists (background)

Local mlx_vlm inference fails on large screenshots two ways, both driven by prefill token count
(which is determined by the model **family's** image processor):
1. **Hard GPU crash** — Metal `std::runtime_error` (GPU Hang / Timeout / InnocentVictim) that
   aborts the whole process (SIGABRT / exit 134). Cannot be caught in-process.
2. **Degenerate output** — runs but emits garbage / empty for the full token budget.

So each model is tested in an **isolated worker subprocess**; when it dies, the parent records the
killing size and continues, so the report is always produced.

## Current state — WORKING

- `benchmark_image_sizes.py` is complete and runs. Parent orchestrator + `--worker` mode.
- Worker: loads model once, tests sizes ascending, computes prefill via `prepare_inputs`
  (no GPU), generates with `WORKER_MAX_TOKENS=128`, classifies, streams `RESULT` JSON, stops at
  first fail. Python exceptions are caught and emitted as `error`/`model_error` (only true C++
  Metal aborts kill the process).
- Parent: subprocess per model, per-size timeout (`PER_SIZE_TIMEOUT_S=240`), distinguishes
  `crash` (Metal signature in stderr) vs `error` (Python traceback) vs `hang` (timeout) vs
  `load_failed`. Stop-at-first-crash with larger sizes marked `skipped (inferred)`.
- Self-contained copy of the mlx_vlm `BPEStreamingDetokenizer.add_token` utf-8 fix
  (`_install_detokenizer_fix`) so degenerate output doesn't re-trigger the detokenizer crash.
- Degeneracy heuristic: empty / low vowel-ratio / low real-word ratio / high repetition /
  ran-to-max-tokens. Validated on good + garbage samples.
- Kimi excluded (`EXCLUDE_MODEL_FILES`) — not cached, missing config.
- 33 models run; 32 of 33 cached with weights (no big downloads needed).

### Verified-good results (reliable rows)
- Gemma-3 / Gemma-4: flat 281 prefill, immune → **3840px**.
- Llama-3.2-Vision (4/8bit): flat 22 prefill (cross-attention) → **3840px**.
- GLM-4.5V/4.6V: coherent at huge prefill → **3840px**.
- Qwen2-VL, Qwen2.5-VL-7B/32B: tolerate 10k+ prefill → **3840px**.
- Qwen2.5-VL-3B: garbage at 1280px → **960px**.
- Qwen3-VL 2B/4B/8B: crash/garbage at 1920px (~2061 prefill) → **1280px**.
- deepseek-vl2-tiny: garbage even at 360px → **none**. deepseek-vl2-small/4bit → 3840px.
- llava 1.5 / v1.6: model/compat **error** (won't run) → **none**.

## ⚠️ OPEN PROBLEM — GPU contamination cascade (must fix before trusting big-model rows)

`Qwen3-VL-32B` **hung** at 1920px (`kIOGPUCommandBufferCallbackErrorTimeout`). That wedged the
**whole GPU**, and every model AFTER it failed with `[METAL] GPU Timeout Error` — even a 2B model
at 480px (109 prefill). Physically impossible as a real size limit (prior run had Qwen3.5/3.6 all
passing 3840px). Subprocess isolation contains Python/RAM but **not a wedged shared GPU**.

**Contaminated/unreliable rows in the last run:** Qwen3-VL-30B-A3B, Qwen3.5-2B/4B/9B/27B/35B,
Qwen3.6-27B/35B.

**Likely partial root cause:** the parent's `proc.kill()` (SIGKILL) fires at the 240s timeout and
terminates the worker **mid-GPU-operation**, which can itself wedge the Metal command queue. The
clean SIGABRT crashes (Qwen3-VL 2B/4B/8B at 1920) did NOT contaminate the next model — only the
**timeout/hang + forced kill** did.

## Fix options (decide + implement next session)

1. **Don't SIGKILL mid-GPU-op** (cheap, high value): SIGTERM + grace before SIGKILL; raise/relax
   per-size timeout for large models (32B prefill is genuinely slow — 240s may kill valid work).
2. **GPU health canary between models** (robust): after any crash/hang, run a trivial inference in
   a throwaway process and poll until it passes before the next model; re-run any model measured
   while the canary was failing. Active recovery-wait beats a blind cooldown.
3. **Predict-and-skip from learned per-family threshold** (elegant + faster): crash size is
   family-consistent (Qwen3-VL 2B/4B/8B all fail at 1920). Learn each family's threshold from its
   smallest model, then for 32B/35B don't even submit larger sizes (mark "predicted-fail"). Never
   triggers the wedging buffer on a giant model. Trade-off: big-model thresholds become inferred.
4. **Reorder smallest-params-first + quarantine giants** (cheap): run hang-prone 32B/35B last (or
   in a separate pass) so a wedge can't poison many rows.
5. **Fail-fast honesty** (safety net): if canary shows GPU wedged and it doesn't recover within N s,
   stop and mark remaining rows "not tested — GPU unstable" instead of emitting false `crash`.

**Recommendation:** implement **1 + 2 + 4** (graceful kill + longer big-model timeout; health
canary with recovery-wait + re-run; smallest-first ordering). Add **3** if a faster run with
inferred big-model thresholds is acceptable. Avoid plain cooldown-retry — it papers over the
SIGKILL issue.

## Files
- `benchmark_image_sizes.py` — the script (parent + `--worker`).
- `images/size/` — the resolution ladder (8 windows-*.jpg, 360→3840).
- `_probe_prefill.py` — per-model prefill counts via processor only (no weights). KEPT (reusable).
- `output/size_matrix_20260618_112227.{json,html}` — authoritative clean matrix (sorted
  provider/family/size) + `_confirm.json` (9/9 failures confirmed real).
- Deleted (one-off repros, no future value): `_probe_size.py`, `_probe_seq.py`.
- Plan: `/Users/pcs/.claude/plans/cached-herding-boole.md`.

## How to run
```bash
python3.11 benchmark_image_sizes.py                      # all models (kimi excluded)
python3.11 benchmark_image_sizes.py --models <id> [<id>] # subset
```
No `.env`/API key needed (no LLM judge in this script).

## Side fixes made this session (in local_benchmark.py context, separate task)
- Root-caused the original 401 (judge key not loaded when run directly — source `.env` first).
- Root-caused the `utf-8` detokenizer crash on Qwen3-VL (mlx_vlm `BPEStreamingDetokenizer.add_token`
  strict decode) — same fix copied into `benchmark_image_sizes.py`.
- Confirmed large screenshots cause GPU faults; `MAX_IMAGE_SIDE` downscaling is the mitigation.
