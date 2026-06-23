#!/usr/bin/env python3
"""Quick single-model / single-image VLM smoke test (mlx_vlm).

Edit MODEL and IMAGE below, then run:  python3.11 quick_test.py
Prints the prefill token count, the model's response, and timing — handy for
checking whether a model survives a given image size before wiring it into the
full benchmark. A hard Metal GPU fault will still abort the process (uncatchable);
that itself is the answer ("this size crashes this model").
"""
import time

# ── Edit these ────────────────────────────────────────────────────────────────
MODEL = "mlx-community/Qwen3-VL-8B-Instruct-8bit"
IMAGE = "images/size/windows-1920.jpg"
PROMPT = "Describe what is shown in this screenshot in one sentence."
MAX_TOKENS = 128
# ──────────────────────────────────────────────────────────────────────────────


def _install_detokenizer_fix():
    """mlx_vlm's BPE detokenizer strict-decodes utf-8 and crashes on the truncated
    multi-byte garbage that degenerate output produces. Re-define add_token with
    errors='replace' so a bad size shows readable garbage instead of aborting."""
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


def main():
    _install_detokenizer_fix()
    from PIL import Image
    from mlx_vlm import load as mlx_load
    from mlx_vlm.generate import generate as mlx_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs

    image = Image.open(IMAGE).convert("RGB")
    print(f"Model: {MODEL}")
    print(f"Image: {IMAGE}  ({image.size[0]}×{image.size[1]})")

    model, processor = mlx_load(MODEL)
    prompt = apply_chat_template(processor, model.config, PROMPT, num_images=1)

    # CPU-side prefill count (governs the size limit) — computed before the GPU pass.
    prefill = int(prepare_inputs(processor, images=image, prompts=prompt)["input_ids"].shape[1])
    print(f"Prefill tokens: {prefill}\n")

    t0 = time.time()
    result = mlx_generate(
        model, processor, prompt=prompt, image=image,
        max_tokens=MAX_TOKENS, temperature=0.0, verbose=False,
    )
    elapsed = time.time() - t0
    text = result.text.split("</think>")[-1].strip()

    print("─" * 60)
    print(text)
    print("─" * 60)
    print(f"{result.generation_tokens} tok | {result.generation_tps:.1f} tok/s | "
          f"{elapsed:.1f}s | peak {result.peak_memory:.2f} GB")


if __name__ == "__main__":
    main()
