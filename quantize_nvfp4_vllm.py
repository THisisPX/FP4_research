"""
Offline NVFP4 W4A16 quantization — vLLM / compressed-tensors 0.15 API.

Tested with compressed-tensors 0.15.0.1 on B300.
Output loads via:  vllm serve ./Qwen3-8B-NVFP4 --quantization compressed-tensors

Usage:
    python3 quantize_nvfp4_vllm.py \
        --model_id Qwen/Qwen3-8B \
        --output_dir ./Qwen3-8B-NVFP4-vllm \
        --device cuda:0
"""

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--output_dir", type=str, default="./Qwen3-8B-NVFP4-vllm")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--group_size", type=int, default=16)
    return parser.parse_args()


# ──────────────────────────────────────────────────────────
#  Use compressed-tensors 0.15 native API
# ──────────────────────────────────────────────────────────

def quantize_linear_weight(weight: torch.Tensor, group_size: int,
                           device: torch.device) -> tuple:
    """
    Returns (weight_packed[uint8], weight_scale[fp8_e4m3fn], weight_global_scale[fp32])
    using the compressed_tensors 0.15 API.
    """
    from compressed_tensors.compressors.nvfp4 import (
        NVFP4PackedCompressor,
    )
    from compressed_tensors.quantization.quant_args import (
        FP4_E2M1_DATA,
        FP8_E4M3_DATA,
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )
    from compressed_tensors.quantization.utils.helpers import generate_gparam

    out_f, in_f = weight.shape
    assert out_f % group_size == 0 and in_f % group_size == 0, \
        f"Dimensions ({out_f},{in_f}) not divisible by group_size {group_size}"

    num_groups = in_f // group_size
    weight_f32 = weight.float().to(device)

    # ---- global scale (fp32, one scalar per row) ----
    global_scale_list = []
    for i in range(out_f):
        amax = torch.max(torch.abs(weight_f32[i]))
        gs = generate_gparam(
            -amax.unsqueeze(0), amax.unsqueeze(0),
            scale_data=FP8_E4M3_DATA, quant_data=FP4_E2M1_DATA,
            dtype=torch.float32,
        )
        global_scale_list.append(gs)
    global_scale = torch.cat(global_scale_list).float()  # (out_f,)

    # ---- block-wise scale (fp8_e4m3fn, one per block) ----
    w = weight_f32.view(out_f, num_groups, group_size)
    block_amax = torch.amax(torch.abs(w), dim=-1)                   # (out_f, num_groups)
    local_scale = block_amax / FP4_E2M1_DATA.max                    # / 6.0
    combined = torch.clamp(
        global_scale.unsqueeze(1) * local_scale,
        min=FP8_E4M3_DATA.min_norm, max=FP8_E4M3_DATA.max,
    )
    weight_scale = combined.to(torch.float8_e4m3fn)                 # (out_f, num_groups)

    # ---- pack via NVFP4PackedCompressor ----
    compressor = NVFP4PackedCompressor()
    quant_args = QuantizationArgs(
        num_bits=4, type=QuantizationType.FLOAT, symmetric=True,
        strategy=QuantizationStrategy.TENSOR_GROUP, group_size=group_size,
        scale_dtype=FP8_E4M3_DATA.dtype,
    )
    result = compressor.compress_weight(
        weight=weight_f32,
        scale=weight_scale.float(),
        global_scale=global_scale.float(),
        quantization_args=quant_args,
    )
    weight_packed = result["weight_packed"].cpu()

    return weight_packed.cpu(), weight_scale.cpu(), global_scale.cpu()


# ──────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────

def should_quantize(name: str, shape: tuple, group_size: int,
                    skip: list[str]) -> bool:
    if not name.endswith(".weight"):
        return False
    if len(shape) != 2:
        return False
    if shape[0] % group_size != 0 or shape[1] % group_size != 0:
        return False
    for p in skip:
        if p in name:
            return False
    return True


def build_vllm_quant_config(group_size: int) -> dict:
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "group",
                    "group_size": group_size,
                    "symmetric": True,
                },
                "input_activations": None,
            }
        },
        "ignore": ["lm_head", "embed_tokens", "re:.*mlp.gate$"],
    }


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("NVFP4 W4A16 Quantization — compressed-tensors 0.15")
    print("=" * 60)
    print(f"Model:       {args.model_id}")
    print(f"Group size:  {args.group_size}")
    print(f"Device:      {args.device}")
    print(f"Output:      {args.output_dir}")

    # ── Step 1: Load ──────────────────────────────────────
    print("\n[1/4] Loading BF16 model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16,
        device_map=args.device, trust_remote_code=True, low_cpu_mem_usage=True,
    )
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  {time.time() - t0:.0f}s  {total_params:.2f}B params  {config.architectures}")

    # ── Step 2: Quantize ──────────────────────────────────
    print(f"\n[2/4] Quantizing Linear weights (compressed-tensors NVFP4PackedCompressor)...")
    t0 = time.time()

    skip = ["lm_head", "model.embed_tokens"]
    new_sd = {}
    quantized = 0
    skipped = 0
    last_layer = -1

    import re
    for name, param in model.named_parameters():
        if should_quantize(name, param.shape, args.group_size, skip):
            try:
                packed, scale, gscale = quantize_linear_weight(
                    param.data, args.group_size, torch.device(args.device),
                )
                base = name.replace(".weight", "")
                new_sd[f"{base}.weight_packed"]       = packed
                new_sd[f"{base}.weight_scale"]        = scale
                new_sd[f"{base}.weight_global_scale"] = gscale
                quantized += 1

                m = re.search(r"layers\.(\d+)\.", name)
                if m:
                    l = int(m.group(1))
                    if l != last_layer:
                        last_layer = l
                        e = time.time() - t0
                        eta = (e / max(l + 1, 1)) * (config.num_hidden_layers - l - 1)
                        print(f"  [{l + 1}/{config.num_hidden_layers}] "
                              f"layer {l}  {e:.0f}s  ETA {eta:.0f}s")
            except Exception as e:
                print(f"  FAILED {name}: {e}")
                new_sd[name] = param.data.cpu()
                skipped += 1
        else:
            new_sd[name] = param.data.cpu()
            skipped += 1

    for name, buf in model.named_buffers():
        if name not in new_sd:
            new_sd[name] = buf.cpu()

    e = time.time() - t0
    print(f"\n  {e:.0f}s — {quantized} quantized  {skipped} skipped")

    # ── Step 3: Save ──────────────────────────────────────
    print(f"\n[3/4] Saving...")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(new_sd, os.path.join(args.output_dir, "pytorch_model.bin"))
    config.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    qc = build_vllm_quant_config(args.group_size)
    with open(os.path.join(args.output_dir, "quantization_config.json"), "w") as f:
        json.dump(qc, f, indent=2)

    # ── Step 4: Verify ────────────────────────────────────
    n_p = sum(1 for k in new_sd if "weight_packed" in k)
    n_s = sum(1 for k in new_sd if "weight_scale" in k)
    n_g = sum(1 for k in new_sd if "weight_global_scale" in k)
    print(f"\n[4/4] Verify:  packed={n_p}  scale={n_s}  global={n_g}")

    if n_p == 0:
        print("ERROR: no quantized weights!"); sys.exit(1)

    sz = sum(t.numel() * t.element_size() for t in new_sd.values()) / 1024 / 1024
    print(f"  BF16: ~{total_params * 2:.0f} GB  →  quantized: {sz / 1024:.1f} GB")

    print(f"\n{'=' * 60}")
    print(f"Done!  {os.path.abspath(args.output_dir)}")
    print(f"{'=' * 60}")
    print(f"\n  vllm serve {os.path.abspath(args.output_dir)} \\")
    print(f"    --trust-remote-code \\")
    print(f"    --quantization compressed-tensors \\")
    print(f"    --dtype bfloat16 --port 8002 \\")
    print(f"    --gpu-memory-utilization 0.90 --max-model-len 4096")


if __name__ == "__main__":
    main()
