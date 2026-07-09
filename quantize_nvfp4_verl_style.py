"""
Offline NVFP4 W4A16 quantization — veRL-style API (standalone, no veRL required).

Uses the same compressed_tensors.NVFP4PackedCompressor that veRL's QATQuantizer
wraps internally, plus veRL's QKV/GateUp scale fusion logic.

Output: checkpoint for  vllm serve --quantization compressed-tensors

Usage (no veRL needed):
    python3 quantize_nvfp4_verl_style.py \
        --model_id Qwen/Qwen3-8B \
        --output_dir ./Qwen3-8B-NVFP4 \
        --device cuda:0
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# ================================================================
#  veRL's QATQuantizer logic, extracted for standalone use
#  Source: verl/utils/qat/quantizer.py
# ================================================================

from compressed_tensors.compressors.nvfp4 import NVFP4PackedCompressor
from compressed_tensors.quantization.quant_args import (
    FP4_E2M1_DATA,
    FP8_E4M3_DATA,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.utils.helpers import generate_gparam

_LAYER_IDX_RE = re.compile(r"layers\.(\d+)\.")

# QKV and GateUp projection fusion patterns (same as veRL)
FUSE_PATTERNS = {
    "qkv": ["q_proj", "k_proj", "v_proj"],
    "gate_up": ["gate_proj", "up_proj"],
}


def compute_blockwise_scale(weight, global_scale, group_size=16):
    """veRL-compatible blockwise scale (FP8 E4M3)."""
    out_f, in_f = weight.shape
    num_groups = in_f // group_size
    w = weight.view(out_f, num_groups, group_size)
    block_max = torch.amax(torch.abs(w), dim=-1).float()
    local_scale = block_max / FP4_E2M1_DATA.max
    combined = torch.clamp(
        global_scale * local_scale,
        min=-FP8_E4M3_DATA.max, max=FP8_E4M3_DATA.max,
    )
    blockwise_scale = combined.to(torch.float8_e4m3fn)
    eps = torch.finfo(torch.float8_e4m3fn).eps
    blockwise_scale = torch.where(
        blockwise_scale == 0,
        torch.full_like(blockwise_scale, eps),
        blockwise_scale,
    )
    return blockwise_scale


def fuse_global_scales(layer_global_scales, strategy="min"):
    """Fuse global scales for QKV/GateUp groups (take min across group)."""
    if not layer_global_scales:
        return {}

    parent_to_children = {}
    for name in layer_global_scales:
        parent, child = name.rsplit(".", 1) if "." in name else ("", name)
        parent_to_children.setdefault(parent, {})[child] = name

    fused_scales = {}
    processed = set()

    for parent, children in parent_to_children.items():
        for _, patterns in FUSE_PATTERNS.items():
            matched = [children[p] for p in patterns if p in children]
            if len(matched) == len(patterns):
                group_scales = [layer_global_scales[n] for n in matched]
                fused_scale = torch.min(torch.cat(group_scales)).reshape([1])
                for layer_name in matched:
                    fused_scales[layer_name] = fused_scale.clone()
                    processed.add(layer_name)

    for name, scale in layer_global_scales.items():
        if name not in processed:
            fused_scales[name] = scale

    return fused_scales


class QATQuantizer:
    """Standalone version of veRL's QATQuantizer.
    No verl dependency — just compressed_tensors + torch."""

    def __init__(self, mode="w4a16", group_size=16,
                 ignore_patterns=None, device=None, param_dtype=None):
        self.mode = mode.lower()
        self._is_w4a4 = self.mode == "w4a4"
        self.group_size = group_size
        self.ignore_patterns = ignore_patterns or [
            "lm_head", "embed_tokens", "re:.*mlp.gate$",
        ]
        self.device = device or torch.device("cuda")
        self.param_dtype = param_dtype

        self._compressor = NVFP4PackedCompressor()
        self._quant_args = QuantizationArgs(
            num_bits=4, type=QuantizationType.FLOAT, symmetric=True,
            strategy=QuantizationStrategy.TENSOR_GROUP, group_size=group_size,
            scale_dtype=FP8_E4M3_DATA.dtype,
        )

    def _should_quantize(self, name, tensor):
        if not name.endswith(".weight"):
            return False
        if tensor.dim() != 2:
            return False
        if tensor.shape[1] % self.group_size != 0:
            return False
        module_name = name.rsplit(".weight", 1)[0]
        for pattern in self.ignore_patterns:
            if pattern.startswith("re:"):
                if re.match(pattern[3:], module_name):
                    return False
            else:
                if pattern in module_name:
                    return False
        return True

    @staticmethod
    def _extract_layer_idx(name):
        m = _LAYER_IDX_RE.search(name)
        return int(m.group(1)) if m else None

    def _process_layer_group(self, layer_idx, layer_params, w4a4_scales, output_device):
        layer_weights = {}
        layer_passthrough = {}

        for name, tensor in layer_params.items():
            if "input_global_scale" in name or "input_amax" in name:
                continue
            if self._should_quantize(name, tensor):
                layer_weights[name.rsplit(".weight", 1)[0]] = (name, tensor)
            else:
                layer_passthrough[name] = tensor

        if layer_idx is None and layer_weights:
            raise RuntimeError(
                f"Quantizable weights outside decoder layers: {list(layer_weights.keys())}"
            )

        if not layer_weights:
            return [(n, t.to(output_device)) for n, t in layer_passthrough.items()]

        # Move to GPU, compute global scales
        weights_on_gpu = {}
        layer_global_scales = {}

        for layer_name, (_, tensor) in layer_weights.items():
            w_gpu = tensor.to(device=self.device, dtype=self.param_dtype)
            weights_on_gpu[layer_name] = w_gpu
            amax = torch.amax(torch.abs(w_gpu)).float()
            layer_global_scales[layer_name] = generate_gparam(
                -amax.unsqueeze(0), amax.unsqueeze(0),
                scale_data=FP8_E4M3_DATA, quant_data=FP4_E2M1_DATA,
                dtype=torch.float32,
            )

        fused = fuse_global_scales(layer_global_scales, strategy="min")
        results = []

        for layer_name, w_gpu in weights_on_gpu.items():
            fgs = fused[layer_name]
            ws = compute_blockwise_scale(w_gpu, fgs, self.group_size)
            wp = self._compressor.compress_weight(
                weight=w_gpu, scale=ws.float(), global_scale=fgs,
                quantization_args=self._quant_args,
            )["weight_packed"]

            results.append((f"{layer_name}.weight_packed", wp.to(output_device)))
            results.append((f"{layer_name}.weight_scale", ws.to(output_device)))
            results.append((f"{layer_name}.weight_global_scale", fgs.to(output_device)))

            if self._is_w4a4:
                if layer_name in w4a4_scales:
                    results.append(
                        (f"{layer_name}.input_global_scale",
                         w4a4_scales[layer_name].float().to(output_device))
                    )
                else:
                    raise ValueError(f"W4A4: missing input_global_scale for {layer_name}")

        del weights_on_gpu, layer_global_scales, fused
        for n, t in layer_passthrough.items():
            results.append((n, t.to(output_device)))
        return results

    def quantize_with_fusion(self, params, target_device=None):
        """Layer-by-layer streaming quantization.
        Yields (name, tensor) pairs."""
        if isinstance(params, dict):
            params = params.items()

        output_device = target_device or torch.device("cpu")
        current_layer_idx = object()
        layer_buffer = {}
        w4a4_scales = {}

        for name, tensor in params:
            t = tensor.to("cpu") if tensor.is_cuda else tensor
            lidx = self._extract_layer_idx(name)

            if self._is_w4a4 and "input_global_scale" in name:
                scale_name = name.replace(".input_global_scale", "")
                if not (t.numel() == 1 and t.item() == -1.0):
                    w4a4_scales[scale_name] = t

            if lidx != current_layer_idx and current_layer_idx is not object() and layer_buffer:
                yield from self._process_layer_group(
                    current_layer_idx, layer_buffer, w4a4_scales, output_device,
                )
                layer_buffer = {}

            current_layer_idx = lidx
            layer_buffer[name] = t

        if layer_buffer:
            yield from self._process_layer_group(
                current_layer_idx, layer_buffer, w4a4_scales, output_device,
            )

        torch.cuda.empty_cache()


# ================================================================
#  Config builder (vLLM compressed-tensors format)
# ================================================================

def build_vllm_quant_config(group_size: int) -> dict:
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4, "type": "float",
                    "strategy": "group", "group_size": group_size,
                    "symmetric": True,
                },
                "input_activations": None,
            }
        },
        "ignore": ["lm_head", "embed_tokens", "re:.*mlp.gate$"],
    }


# ================================================================
#  Main
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--output_dir", type=str, default="./Qwen3-8B-NVFP4")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--group_size", type=int, default=16)
    args = parser.parse_args()

    print("=" * 60)
    print("NVFP4 W4A16 Quantization — veRL-style (standalone)")
    print("=" * 60)
    print(f"Model:       {args.model_id}")
    print(f"Group size:  {args.group_size}")
    print(f"Device:      {args.device}")
    print(f"Output:      {args.output_dir}")

    # ── 1. Load ────────────────────────────────────────────
    print("\n[1/4] Loading BF16 model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16,
        device_map=args.device, trust_remote_code=True, low_cpu_mem_usage=True,
    )
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    total = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  {time.time() - t0:.0f}s  {total:.2f}B params  arch={config.architectures}")

    # ── 2. Quantize ────────────────────────────────────────
    print(f"\n[2/4] Quantizing (veRL-style QATQuantizer)...")
    t0 = time.time()

    q = QATQuantizer(
        mode="w4a16", group_size=args.group_size,
        device=torch.device(args.device), param_dtype=torch.bfloat16,
    )
    params = dict(model.named_parameters())
    new_sd = {}
    last_layer = -1

    for name, tensor in q.quantize_with_fusion(params, target_device=torch.device("cpu")):
        new_sd[name] = tensor
        m = re.search(r"layers\.(\d+)\.", name)
        if m:
            l = int(m.group(1))
            if l != last_layer:
                last_layer = l
                e = time.time() - t0
                eta = (e / max(l + 1, 1)) * (config.num_hidden_layers - l - 1)
                print(f"  [{l + 1}/{config.num_hidden_layers}] "
                      f"layer {l}  {e:.0f}s  ETA {eta:.0f}s")

    e = time.time() - t0
    print(f"\n  Done in {e:.0f}s  entries={len(new_sd)}")

    # ── 3. Save ────────────────────────────────────────────
    print(f"\n[3/4] Saving...")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(new_sd, os.path.join(args.output_dir, "pytorch_model.bin"))
    config.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    qc = build_vllm_quant_config(args.group_size)
    with open(os.path.join(args.output_dir, "quantization_config.json"), "w") as f:
        json.dump(qc, f, indent=2)

    # ── 4. Verify ──────────────────────────────────────────
    n_p = sum(1 for k in new_sd if "weight_packed" in k)
    n_s = sum(1 for k in new_sd if "weight_scale" in k)
    n_g = sum(1 for k in new_sd if "weight_global_scale" in k)
    sz = sum(t.numel() * t.element_size() for t in new_sd.values()) / 1024 / 1024
    print(f"\n[4/4] Verify:  packed={n_p}  scale={n_s}  global={n_g}")
    print(f"  BF16 ~{total * 2:.0f} GB  →  Quantized {sz / 1024:.1f} GB")

    if n_p == 0:
        print("ERROR: no quantized weights!"); sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"Done!  {os.path.abspath(args.output_dir)}")
    print(f"{'=' * 60}")
    print(f"\n  vllm serve {os.path.abspath(args.output_dir)} \\")
    print(f"    --trust-remote-code --quantization compressed-tensors \\")
    print(f"    --dtype bfloat16 --port 8002 \\")
    print(f"    --gpu-memory-utilization 0.90 --max-model-len 4096")


if __name__ == "__main__":
    main()
