"""Report prefill token counts per test image, per model — WITHOUT downloading
model weights or running the GPU. Uses only the processor + config (small files)
and mlx_vlm's prepare_inputs (the same CPU-side input prep generate() uses).

Usage: python3.11 _probe_prefill.py [model_files...]   (default: all local_*_models.json)
"""
import glob
import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from mlx_vlm.utils import load_processor, load_config, prepare_inputs
from mlx_vlm.prompt_utils import apply_chat_template

from local_benchmark import load_image, TEST_CASES_FILE  # reuse resize + cases

# Only config/tokenizer/processor files — NO *.safetensors (skip multi-GB weights)
_NO_WEIGHTS = ["*.json", "*.py", "*.model", "*.tiktoken", "*.txt", "*.jinja"]


def resolve(mid):
    return Path(snapshot_download(repo_id=mid, allow_patterns=_NO_WEIGHTS))

model_files = sys.argv[1:] or sorted(
    f for f in glob.glob("local_*_models.json") if "test" not in f
)

# Collect unique model ids (preserve order)
model_ids = []
for mf in model_files:
    for entry in json.load(open(mf)):
        mid = entry["name"] if isinstance(entry, dict) else entry
        if mid not in model_ids:
            model_ids.append(mid)

tests = json.load(open(TEST_CASES_FILE))
images = []
for c in tests:
    p = c.get("image", "")
    if p and p not in images:
        images.append(p)

q = "Describe this screenshot."  # generic; prefill is dominated by image tokens
print(f"Models: {len(model_ids)} | Images: {len(images)}\n")

for mid in model_ids:
    print(f"=== {mid} ===")
    try:
        path = resolve(mid)
        cfg = load_config(path)
        proc = load_processor(path)
    except Exception as e:
        print(f"  load failed: {type(e).__name__}: {e}\n")
        continue
    for p in images:
        try:
            im = load_image(p)
            prompt = apply_chat_template(proc, cfg, q, num_images=1)
            inputs = prepare_inputs(proc, images=im, prompts=prompt)
            n = int(inputs["input_ids"].shape[1])
            print(f"  {n:6d}  {os.path.basename(p)}  (img {im.size})")
        except Exception as e:
            print(f"  ERROR   {os.path.basename(p)}: {type(e).__name__}: {e}")
    print()
