#!/usr/bin/env python3
"""
Image-size stability matrix for local VLMs (mlx_vlm).

Runs each model against a ladder of screenshot resolutions and reports, per model,
which image sizes are SUPPORTED, produce DEGENERATE output, or CRASH the GPU — plus
each model's max supported resolution.

Why a subprocess-per-model design? Oversized images fault Apple's Metal GPU with a
hard `std::runtime_error` (GPU Hang / InnocentVictim) that aborts the *whole* Python
process (SIGABRT / exit 134) — it cannot be caught with try/except. So each model runs
in an isolated worker; when a worker dies, the parent records the killing size as a
crash and keeps going, so the report is always produced.

Usage:
    python3.11 benchmark_image_sizes.py                     # all local_*_models.json
    python3.11 benchmark_image_sizes.py --models <id> [<id> ...]
    python3.11 benchmark_image_sizes.py --worker <id>       # internal (one model)
"""

import argparse
import glob
import json
import os
import re
import select
import subprocess
import sys
import time

# ----------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------
IMAGE_SIZE_DIR = "images/size"
MODEL_FILES_GLOB = "local_*_models.json"
EXCLUDE_MODEL_FILES = {"local_test_models.json", "local_kimi_models.json"}
WORKER_MAX_TOKENS = 128
PER_SIZE_TIMEOUT_S = 240          # generous: large models prefill slowly
LARGE_MODEL_TIMEOUT_S = 600       # even more headroom for big models (avoid false-killing slow ones)
LARGE_MODEL_B = 20.0             # >= this many B params counts as "large"
TERMINATE_GRACE_S = 12           # SIGTERM grace before SIGKILL (avoid killing mid-GPU-op abruptly)
PROMPT = "Describe what is shown in this screenshot in one sentence."

# GPU-recovery (option 2): a hard hang can wedge the whole Metal GPU, poisoning later
# models. After any crash/hang we verify the GPU works again with a tiny model-free matmul
# (canary) and wait for recovery before continuing; a crash at small prefill is treated as
# likely contamination and the model is re-run once on a verified-healthy GPU.
CANARY_TIMEOUT_S = 60
RECOVERY_POLL_S = 15
MAX_RECOVERY_WAIT_S = 240
SUSPECT_CRASH_PREFILL = 1500     # a crash/hang below this prefill is implausible as a real
                                 # size limit → assume GPU contamination and retry once
_CANARY_CODE = (
    "import mlx.core as mx; a = mx.random.normal((2048, 2048)); "
    "mx.eval((a @ a).sum())"
)

# Degeneracy thresholds (no LLM judge — pure heuristics over the response text)
DEGEN_MIN_WORD_RATIO = 0.5        # fraction of whitespace tokens that are real words
DEGEN_MAX_REPEAT_RATIO = 0.5      # fraction of tokens taken by the single most common one
DEGEN_MIN_TOKENS_TO_JUDGE = 8     # very short replies aren't judged as gibberish

STATUS_PASS = "pass"
STATUS_DEGENERATE = "degenerate"
STATUS_CRASH = "crash"
STATUS_HANG = "hang"
STATUS_SKIPPED = "skipped"        # inferred-fail (a smaller size already failed)
STATUS_LOAD_FAILED = "load_failed"
STATUS_ERROR = "error"            # worker died from a Python exception, not a GPU fault

# stderr signatures of a real Apple Metal GPU fault (vs. an ordinary Python traceback)
_METAL_SIGNATURE = re.compile(
    r"\[METAL\]|Metal|GPU Hang|kIOGPU|InnocentVictim|command buffer", re.I
)


# ----------------------------------------------------------------------------------
# mlx_vlm detokenizer fix (self-contained copy)
#
# mlx_vlm's BPEStreamingDetokenizer.add_token does a strict `.decode("utf-8")` that
# crashes on truncated multi-byte sequences — exactly what degenerate output produces.
# Without this, a "degenerate" case would crash instead of being classified. Re-define
# add_token identically but with errors="replace".
# ----------------------------------------------------------------------------------
def _install_detokenizer_fix():
    from mlx_vlm import tokenizer_utils as _t

    def _safe_add_token(self, token, skip_special_token_ids=[]):
        if token in skip_special_token_ids:
            return
        v = self.tokenmap[token]
        if self._byte_decoder[v[0]] == 32:
            current_text = bytearray(
                self._byte_decoder[c] for c in self._unflushed
            ).decode("utf-8", errors="replace")
            if self.text or not self.trim_space:
                self.text += current_text
            else:
                self.text += _t._remove_space(current_text)
            self._unflushed = v
        else:
            self._unflushed += v

    _t.BPEStreamingDetokenizer.add_token = _safe_add_token


# ----------------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------------
SIZE_FILTER = None   # set[int] | None — restrict which image sizes are tested (--sizes)


def discover_sizes():
    """Return [(size_px, path), ...] sorted ascending by longest edge.

    Size is parsed from the filename suffix (windows-1920.jpg -> 1920); falls back to
    the image's actual longest edge if no numeric suffix is present. Honors SIZE_FILTER
    so a single (model, size) can be tested in isolation.
    """
    from PIL import Image

    entries = []
    for p in sorted(glob.glob(os.path.join(IMAGE_SIZE_DIR, "*"))):
        if not p.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        m = re.search(r"(\d{3,4})(?=\.[^.]+$)", os.path.basename(p))
        size = int(m.group(1)) if m else max(Image.open(p).size)
        entries.append((size, p))
    if SIZE_FILTER is not None:
        entries = [e for e in entries if e[0] in SIZE_FILTER]
    entries.sort(key=lambda e: e[0])
    return entries


def provider_key_from_file(models_file):
    base = os.path.splitext(os.path.basename(models_file))[0]
    return base.replace("local_", "", 1).replace("_models", "")


def provider_display(key):
    return key.capitalize() if key else "Other"


def discover_models():
    """Return [(model_id, provider_display), ...] from local_*_models.json (deduped)."""
    seen = set()
    models = []
    for mf in sorted(glob.glob(MODEL_FILES_GLOB)):
        if os.path.basename(mf) in EXCLUDE_MODEL_FILES:
            continue
        provider = provider_display(provider_key_from_file(mf))
        for entry in json.load(open(mf)):
            mid = entry["name"] if isinstance(entry, dict) else entry
            if mid not in seen:
                seen.add(mid)
                models.append((mid, provider))
    return models


def _is_real_word(token):
    """Cheap 'looks like a real word' test, used to spot consonant-soup gibberish.

    Real English words have a vowel, aren't absurdly long, and aren't long all-caps
    runs (which catch garbage like 'MCTMs', 'TCIDIDTCMSMSICFMSFS', 'BCFTSCSCMCTIMS').
    Punctuation/number-only tokens return None (neutral — excluded from the ratio).
    """
    letters = re.sub(r"[^A-Za-z]", "", token)
    if not letters:
        return None
    if len(letters) > 16:
        return False
    if not re.search(r"[aeiouAEIOU]", letters):
        return False
    if len(letters) >= 5 and letters.isupper():
        return False
    return True


def classify_response(text, output_tokens, max_tokens):
    """Return (status, reason) — STATUS_PASS or STATUS_DEGENERATE."""
    stripped = (text or "").strip()
    if not stripped:
        return STATUS_DEGENERATE, "empty response"

    # Global vowel ratio over alphabetic characters (English ≈ 0.38; gibberish far lower).
    alpha = re.sub(r"[^A-Za-z]", "", stripped)
    if alpha:
        vowel_ratio = sum(c in "aeiouAEIOU" for c in alpha) / len(alpha)
    else:
        vowel_ratio = 0.0

    tokens = stripped.split()
    if len(tokens) < DEGEN_MIN_TOKENS_TO_JUDGE:
        # Short but non-empty — accept unless it's clearly non-verbal junk.
        if alpha and vowel_ratio >= 0.2 and re.search(r"[A-Za-z]{2,}", stripped):
            return STATUS_PASS, ""
        return STATUS_DEGENERATE, "short non-verbal response"

    judged = [_is_real_word(t) for t in tokens]
    judged = [j for j in judged if j is not None]
    word_ratio = (sum(judged) / len(judged)) if judged else 0.0

    counts = {}
    for w in tokens:
        counts[w] = counts.get(w, 0) + 1
    repeat_ratio = max(counts.values()) / len(tokens)

    if vowel_ratio < 0.26:
        return STATUS_DEGENERATE, f"low vowel ratio {vowel_ratio:.2f}"
    if word_ratio < DEGEN_MIN_WORD_RATIO:
        return STATUS_DEGENERATE, f"low real-word ratio {word_ratio:.2f}"
    if repeat_ratio > DEGEN_MAX_REPEAT_RATIO:
        return STATUS_DEGENERATE, f"high repetition {repeat_ratio:.2f}"
    if output_tokens >= max_tokens and word_ratio < 0.7:
        return STATUS_DEGENERATE, "ran to max_tokens without stopping"
    return STATUS_PASS, ""


# ----------------------------------------------------------------------------------
# Worker: load one model, test sizes ascending, stream results, stop at first fail.
# ----------------------------------------------------------------------------------
def run_worker(model_id):
    _install_detokenizer_fix()
    from PIL import Image
    from mlx_vlm import load as mlx_load
    from mlx_vlm.generate import generate as mlx_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs

    def emit(obj):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    try:
        model, processor = mlx_load(model_id)
    except Exception as e:
        emit({"type": "load_failed", "error": f"{type(e).__name__}: {e}"})
        return

    # Prompt building can itself raise on incompatible processors (e.g. some llava
    # builds) — treat that as a whole-model error, not a per-size crash.
    try:
        prompt = apply_chat_template(processor, model.config, PROMPT, num_images=1)
    except Exception as e:
        emit({"type": "model_error", "error": f"{type(e).__name__}: {e}"})
        return

    for size, path in discover_sizes():
        image = Image.open(path).convert("RGB")
        # Count prefill tokens BEFORE the GPU forward pass, so a size that then
        # crashes still has its prefill recorded in the START line.
        try:
            prefill = int(prepare_inputs(processor, images=image, prompts=prompt)["input_ids"].shape[1])
        except Exception:
            prefill = -1
        emit({"type": "start", "size": size, "prefill": prefill})

        # A Python exception here = model/compat error (caught, reported, stop). Only an
        # uncatchable C++ Metal abort kills the process — the parent infers that as a crash.
        start = time.time()
        try:
            result = mlx_generate(
                model, processor, prompt=prompt, image=image,
                max_tokens=WORKER_MAX_TOKENS, temperature=0.0, verbose=False,
            )
        except Exception as e:
            emit({"type": "result", "size": size, "status": STATUS_ERROR,
                  "reason": f"{type(e).__name__}: {e}", "prefill": prefill,
                  "output_tokens": 0, "tps": 0, "peak_mem": 0, "time": 0,
                  "snippet": f"{type(e).__name__}: {e}"[:200]})
            return
        elapsed = time.time() - start
        text = result.text.split("</think>")[-1].strip()
        status, reason = classify_response(text, result.generation_tokens, WORKER_MAX_TOKENS)

        emit({
            "type": "result", "size": size, "status": status, "reason": reason,
            "prefill": prefill, "output_tokens": result.generation_tokens,
            "tps": round(result.generation_tps, 1),
            "peak_mem": round(result.peak_memory, 2),
            "time": round(elapsed, 2), "snippet": text[:200],
        })

        if status != STATUS_PASS:
            # Monotonic assumption: bigger only gets worse. Stop here (clean exit);
            # the parent marks larger sizes as skipped/inferred.
            return


# ----------------------------------------------------------------------------------
# Parent: orchestrate one isolated worker per model.
# ----------------------------------------------------------------------------------
def read_line_with_timeout(proc, timeout):
    """Read one line from proc.stdout, or return None if `timeout` s pass with no output."""
    fd = proc.stdout.fileno()
    rlist, _, _ = select.select([fd], [], [], timeout)
    if not rlist:
        return None
    line = proc.stdout.readline()
    return line if line else ""   # "" => EOF


def param_billions(model_id):
    """Best-effort parameter count (in billions) parsed from the model id, for ordering
    and timeout scaling. 'tiny'/'small' map to small sizes; '32b'/'30b-a3b' -> 30/32 (MoE
    total footprint); the 4bit/8bit quant suffix is ignored; unknown -> mid-pack."""
    n = model_id.lower()
    if "tiny" in n:
        return 1.0
    if "small" in n:
        return 3.0
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*b\b", n)   # 2b, 32b, e2b->2b ; '4bit' won't match
    if matches:
        return max(float(m) for m in matches)
    return 12.0


def family_key(model_id):
    """Model family (series without the size), e.g. 'Qwen3-VL-2B-Instruct-4bit' -> 'Qwen3-VL',
    'gemma-4-26b-a4b-it-4bit' -> 'gemma-4'. Tokens up to (not including) the first size token
    (2B/e2b/A3B/7b/tiny/small)."""
    name = re.sub(r"-\d+bit$", "", model_id.split("/")[-1])   # drop -4bit/-8bit quant suffix
    fam = []
    for p in name.split("-"):
        if re.fullmatch(r"[eEaA]?\d+(?:\.\d+)?[bB]", p) or p.lower() in ("tiny", "small"):
            break
        fam.append(p)
    return "-".join(fam) if fam else name


def timeout_for(model_id):
    return LARGE_MODEL_TIMEOUT_S if param_billions(model_id) >= LARGE_MODEL_B else PER_SIZE_TIMEOUT_S


def terminate(proc):
    """Graceful stop: SIGTERM + grace, then SIGKILL only if still alive. Abruptly SIGKILLing
    a worker mid-GPU-operation can itself wedge the Metal command queue, so we give it a
    chance to unwind first."""
    proc.terminate()
    try:
        proc.wait(timeout=TERMINATE_GRACE_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def gpu_healthy():
    """Run a tiny model-free Metal matmul in a throwaway process. A wedged GPU makes it
    abort/timeout; returns True only if it completes cleanly."""
    try:
        r = subprocess.run([sys.executable, "-c", _CANARY_CODE],
                           capture_output=True, text=True, timeout=CANARY_TIMEOUT_S)
        return r.returncode == 0 and not _METAL_SIGNATURE.search(r.stderr or "")
    except subprocess.TimeoutExpired:
        return False


def wait_for_gpu():
    """Poll the canary until the GPU recovers (or give up after MAX_RECOVERY_WAIT_S)."""
    waited = 0
    while True:
        if gpu_healthy():
            return True
        if waited >= MAX_RECOVERY_WAIT_S:
            return False
        time.sleep(RECOVERY_POLL_S)
        waited += RECOVERY_POLL_S


def suspect_contamination(cells):
    """True if any crash/hang happened at an implausibly small prefill — almost certainly a
    wedged GPU from a prior model, not a real size limit for this one."""
    for c in cells.values():
        if c.get("status") in (STATUS_CRASH, STATUS_HANG):
            pf = c.get("prefill")
            if pf is not None and 0 <= pf < SUSPECT_CRASH_PREFILL:
                return True
    return False


def had_gpu_fault(cells):
    return any(c.get("status") in (STATUS_CRASH, STATUS_HANG) for c in cells.values())


def run_model(model_id, sizes):
    """Run a worker for one model; return {size: record} for every size."""
    cells = {}
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", model_id]
    if SIZE_FILTER is not None:
        cmd += ["--sizes", ",".join(str(s) for s in sorted(SIZE_FILTER))]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    pending_size = None        # size announced via START but not yet RESULTed
    pending_prefill = None
    load_failed = False
    model_error = None         # whole-model Python error (e.g. prompt build failed)
    timed_out = False
    timeout = timeout_for(model_id)

    while True:
        line = read_line_with_timeout(proc, timeout)
        if line is None:                      # timeout — treat as GPU hang
            timed_out = True
            terminate(proc)                   # graceful: SIGTERM + grace, then SIGKILL
            break
        if line == "":                        # EOF — worker exited
            break
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue                          # ignore stray stdout (progress bars etc.)

        if msg["type"] == "load_failed":
            load_failed = True
        elif msg["type"] == "model_error":
            model_error = msg.get("error", "")
        elif msg["type"] == "start":
            pending_size = msg["size"]
            pending_prefill = msg.get("prefill")
        elif msg["type"] == "result":
            cells[msg["size"]] = msg
            pending_size = None

    proc.wait()
    stderr_full = proc.stderr.read() or ""
    stderr_tail = stderr_full[-600:]

    if load_failed:
        for size, _ in sizes:
            cells[size] = {"status": STATUS_LOAD_FAILED, "size": size}
        return cells

    if model_error is not None:
        for size, _ in sizes:
            cells[size] = {"status": STATUS_ERROR, "size": size, "snippet": model_error}
        return cells

    # A START with no matching RESULT means the worker died on that size with an
    # *uncatchable* fault. Tell a real Metal GPU abort (Metal signature in stderr) apart
    # from a Python crash (Traceback) so the matrix doesn't mislabel the cause. Search
    # the full stderr — "Traceback" sits at the top, often beyond a short tail.
    if pending_size is not None and pending_size not in cells:
        if timed_out:
            status = STATUS_HANG
        elif _METAL_SIGNATURE.search(stderr_full):
            status = STATUS_CRASH
        elif "Traceback" in stderr_full:
            status = STATUS_ERROR
        else:
            status = STATUS_CRASH
        cells[pending_size] = {
            "status": status, "size": pending_size, "prefill": pending_prefill,
            "snippet": stderr_tail.strip(),
        }

    # Fill any untested larger sizes as inferred-skip once a failure occurred.
    failed = any(c.get("status") in (STATUS_CRASH, STATUS_HANG, STATUS_DEGENERATE, STATUS_ERROR)
                 for c in cells.values())
    if failed:
        for size, _ in sizes:
            cells.setdefault(size, {"status": STATUS_SKIPPED, "size": size})
    return cells


def max_supported(cells, sizes):
    """Largest size in the contiguous PASS run starting from the smallest."""
    best = None
    for size, _ in sizes:
        if cells.get(size, {}).get("status") == STATUS_PASS:
            best = size
        else:
            break
    return best


# ----------------------------------------------------------------------------------
# HTML report
# ----------------------------------------------------------------------------------
_STATUS_STYLE = {
    STATUS_PASS:        ("#dcfce7", "#166534", "✓"),
    STATUS_DEGENERATE:  ("#fef9c3", "#854d0e", "≈ garbage"),
    STATUS_CRASH:       ("#fee2e2", "#991b1b", "✗ GPU crash"),
    STATUS_HANG:        ("#fee2e2", "#991b1b", "✗ GPU hang"),
    STATUS_ERROR:       ("#ede9fe", "#5b21b6", "⚠ error"),
    STATUS_SKIPPED:     ("#f1f5f9", "#64748b", "– skipped"),
    STATUS_LOAD_FAILED: ("#e2e8f0", "#334155", "load failed"),
}


def generate_html_report(rows, sizes, output_path):
    import html
    from PIL import Image

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Thumbnail + label per size column
    head_cells = ""
    for size, path in sizes:
        try:
            w, h = Image.open(path).size
            dims = f"{w}×{h}"
        except Exception:
            dims = ""
        head_cells += f'<th>{size}px<br><span style="font-weight:400;color:#64748b;font-size:0.75rem;">{dims}</span></th>'

    # Sort by provider, then family, then param size (smallest->largest), then name.
    rows = sorted(
        rows,
        key=lambda r: (r.get("provider", "").lower(), family_key(r["model"]).lower(),
                       param_billions(r["model"]), r["model"]),
    )
    ncols = len(sizes) + 2   # model + sizes + max-supported

    body = ""
    cur_provider = None
    for r in rows:
        provider = r.get("provider", "Other")
        if provider != cur_provider:
            body += f"""
        <tr class="provider-row"><td colspan="{ncols}">{html.escape(provider)}</td></tr>"""
            cur_provider = provider

        cells_html = ""
        for size, _ in sizes:
            c = r["cells"].get(size, {"status": STATUS_SKIPPED})
            bg, fg, label = _STATUS_STYLE.get(c["status"], ("#fff", "#000", c["status"]))
            tip = []
            if c.get("prefill") not in (None, -1):
                tip.append(f"prefill {c['prefill']} tok")
            if c.get("reason"):
                tip.append(c["reason"])
            if c.get("snippet"):
                tip.append(c["snippet"])
            title = html.escape(" | ".join(tip))
            cells_html += (
                f'<td style="background:{bg};color:{fg};font-weight:600;text-align:center;" '
                f'title="{title}">{label}</td>'
            )
        ms = r["max_supported"]
        ms_html = f'{ms}px' if ms else '<span style="color:#991b1b;">none</span>'
        body += f"""
        <tr>
          <td style="font-weight:600;padding-left:1.5rem;">{html.escape(r['model'].split('/')[-1])}<br>
            <span style="font-weight:400;color:#64748b;font-size:0.78rem;">{html.escape(family_key(r['model']))}</span></td>
          {cells_html}
          <td style="text-align:center;font-weight:700;">{ms_html}</td>
        </tr>"""

    legend = "".join(
        f'<span style="background:{bg};color:{fg};padding:0.2rem 0.6rem;border-radius:6px;'
        f'margin-right:0.5rem;font-size:0.8rem;font-weight:600;">{label}</span>'
        for st, (bg, fg, label) in _STATUS_STYLE.items()
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>VLM Image-Size Stability Matrix - {timestamp}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#f8fafc; color:#1e293b; padding:2rem; }}
  .container {{ max-width:1400px; margin:0 auto; }}
  h1 {{ font-size:1.7rem; margin-bottom:0.4rem; }}
  .timestamp {{ color:#64748b; margin-bottom:1rem; font-size:0.9rem; }}
  .legend {{ margin-bottom:1.5rem; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:12px; overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,0.1); font-size:0.88rem; }}
  th {{ background:#f1f5f9; padding:0.7rem 0.5rem; text-align:center; font-weight:600; color:#475569; border-bottom:2px solid #e2e8f0; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:0.6rem 0.5rem; border-bottom:1px solid #f1f5f9; }}
  .provider-row td {{ background:#eef2ff; color:#3730a3; font-weight:700; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.05em; border-bottom:2px solid #c7d2fe; }}
</style></head>
<body><div class="container">
  <h1>🖥️ VLM Image-Size Stability Matrix</h1>
  <p class="timestamp">Generated: {timestamp} | Models: {len(rows)} | Sizes: {len(sizes)} | Prompt: "{html.escape(PROMPT)}"</p>
  <div class="legend">{legend}</div>
  <table>
    <thead><tr><th>Model</th>{head_cells}<th>Max supported</th></tr></thead>
    <tbody>{body}
    </tbody>
  </table>
</div></body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


_FAIL_STATUSES = (STATUS_CRASH, STATUS_HANG, STATUS_DEGENERATE, STATUS_ERROR)


def confirm_failures(matrix_path):
    """Re-run each model's first observed failure at ONLY that size, in a fresh process,
    so nothing runs before it. If it still fails => real limit. If it now passes => the
    original failure was contamination from a prior run."""
    global SIZE_FILTER
    d = json.load(open(matrix_path))
    sizes = discover_sizes()                       # SIZE_FILTER is None here => all sizes
    size_path = {s: p for s, p in sizes}

    # One observed failure per model (stop-at-first-fail => first non-pass real status).
    todo = []
    for r in d["rows"]:
        for s, _ in sizes:
            st = (r["cells"].get(str(s)) or r["cells"].get(s) or {}).get("status")
            if st in _FAIL_STATUSES:
                todo.append((r["model"], s, st))
                break

    print(f"Confirming {len(todo)} failed test case(s) individually (isolated, single-size):\n",
          flush=True)
    results = []
    gpu_suspect = False
    for model_id, size, orig in todo:
        if gpu_suspect:
            wait_for_gpu()
            gpu_suspect = False
        SIZE_FILTER = {size}
        cells = run_model(model_id, [(size, size_path[size])])
        SIZE_FILTER = None
        new = cells.get(size, {}).get("status", "?")
        if new in (STATUS_CRASH, STATUS_HANG):
            gpu_suspect = True
        confirmed = new in _FAIL_STATUSES
        verdict = "CONFIRMED real" if confirmed else "NOT reproduced (was contamination!)"
        results.append({"model": model_id, "size": size, "original": orig,
                        "isolated": new, "confirmed": confirmed})
        print(f"  {model_id.split('/')[-1]:<34} {size:>4}px  "
              f"orig={orig:<10} isolated={new:<10} -> {verdict}", flush=True)

    n_ok = sum(r["confirmed"] for r in results)
    print(f"\n{n_ok}/{len(results)} failures confirmed real; "
          f"{len(results)-n_ok} did NOT reproduce in isolation.", flush=True)

    out = matrix_path.replace(".json", "_confirm.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Confirmation saved to: {out}", flush=True)


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="VLM image-size stability matrix")
    parser.add_argument("--worker", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific model id(s) to test (default: all local_*_models.json)")
    parser.add_argument("--sizes", type=str, default=None,
                        help="Comma-separated px sizes to test (default: all in images/size/)")
    parser.add_argument("--confirm", nargs="?", const="__latest__", default=None,
                        help="Re-run each failed cell of a matrix JSON in isolation to confirm "
                             "the failure is real (default: latest output/size_matrix_*.json)")
    args = parser.parse_args()

    global SIZE_FILTER
    if args.sizes:
        SIZE_FILTER = {int(s) for s in args.sizes.split(",") if s.strip()}

    if args.worker:
        run_worker(args.worker)
        return

    if args.confirm is not None:
        path = args.confirm
        if path == "__latest__":
            cands = glob.glob(os.path.join("output", "size_matrix_*.json"))
            cands = [c for c in cands if "_confirm" not in c]
            if not cands:
                print("ERROR: no matrix JSON found to confirm.")
                sys.exit(1)
            path = max(cands, key=os.path.getmtime)
        confirm_failures(path)
        return

    sizes = discover_sizes()
    if not sizes:
        print(f"ERROR: no images found in {IMAGE_SIZE_DIR}/")
        sys.exit(1)

    if args.models:
        models = [(m, provider_display(next((provider_key_from_file(f)
                   for f in glob.glob(MODEL_FILES_GLOB)
                   if m in [e.get("name") if isinstance(e, dict) else e for e in json.load(open(f))]),
                   "other"))) for m in args.models]
    else:
        models = discover_models()

    # Option 4: smallest params first, so hang-prone giants run last and can't poison
    # the rest of the run if they wedge the GPU.
    models.sort(key=lambda mp: param_billions(mp[0]))

    print("=" * 70)
    print("  IMAGE-SIZE STABILITY MATRIX")
    print(f"  Models: {len(models)} | Sizes: {[s for s, _ in sizes]}")
    print("=" * 70, flush=True)

    rows = []
    gpu_suspect = False        # set after any crash/hang: verify GPU before the next model
    for i, (model_id, provider) in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {model_id} ({provider})", flush=True)

        # Option 2: if a prior model may have wedged the GPU, wait for it to recover first.
        if gpu_suspect:
            print("    checking GPU health before starting...", flush=True)
            if not wait_for_gpu():
                print("    ⚠️  GPU did not recover; results below may be unreliable.", flush=True)
            gpu_suspect = False

        cells = run_model(model_id, sizes)

        # Option 2 (retry): a crash/hang at a small prefill is almost certainly leftover GPU
        # contamination, not a real limit for this model — recover and re-run once.
        if suspect_contamination(cells):
            print("    suspicious crash at small prefill — recovering GPU and retrying once...", flush=True)
            wait_for_gpu()
            retry = run_model(model_id, sizes)
            if not suspect_contamination(retry):
                cells = retry
            elif max_supported(retry, sizes) and not max_supported(cells, sizes):
                cells = retry

        if had_gpu_fault(cells):
            gpu_suspect = True

        for size, _ in sizes:
            st = cells.get(size, {}).get("status", "?")
            print(f"    {size:>4}px : {st}")
        ms = max_supported(cells, sizes)
        print(f"    -> max supported: {ms}px" if ms else "    -> max supported: none", flush=True)
        rows.append({"model": model_id, "provider": provider,
                     "cells": cells, "max_supported": ms})

    os.makedirs("output", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join("output", f"size_matrix_{ts}.json")
    html_path = os.path.join("output", f"size_matrix_{ts}.html")
    with open(json_path, "w") as f:
        json.dump({"sizes": [s for s, _ in sizes], "rows": rows}, f, indent=2)
    generate_html_report(rows, sizes, html_path)

    print(f"\nJSON saved to: {json_path}")
    print(f"HTML report:   {html_path}")


if __name__ == "__main__":
    main()
