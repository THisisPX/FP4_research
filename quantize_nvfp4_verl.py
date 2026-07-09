"""
Script A: Offline NVFP4 W4A16 quantization using veRL's QATQuantizer API.

Relies on: verl.utils.qat.QATQuantizer + compressed_tensors.NVFP4PackedCompressor

Output: a checkpoint that vLLM can serve with:
    vllm serve ./Qwen3-8B-NVFP4-verl --trust-remote-code \\
        --quantization compressed-tensors --port 8002

Usage:
    python quantize_nvfp4_verl.py --model_id Qwen/Qwen3-8B --output_dir ./Qwen3-8B-NVFP4-verl [--w4a4]
"""

import argparse
import json
import os
import shutil
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline NVFP4 quantization using veRL's QATQuantizer"
    )
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--output_dir", type=str, default="./Qwen3-8B-NVFP4-verl")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--mode", type=str, default="w4a16", choices=["w4a16", "w4a4"])
    parser.add_argument("--verl_path", type=str, default="D:/learning/verl_cambricon")
    return parser.parse_args()


def build_quantization_config_json(group_size: int, mode: str) -> dict:
    """Build the quantization_config.json that vLLM's compressed-tensors
    backend expects."""
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "global_compression_ratio": 3.92,
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "group",
                    "group_size": group_size,
                    "symmetric": True,
                    "actorder": False,
                },
                "input_activations": (
                    None if mode == "w4a16"
                    else {
                        "num_bits": 4,
                        "type": "float",
                        "strategy": "group",
                        "group_size": group_size,
                        "symmetric": True,
                    }
                ),
            }
        },
    }


def main():
    args = parse_args()

    # Add veRL to Python path
    sys.path.insert(0, args.verl_path)

    # Import veRL's quantizer after path setup
    from verl.utils.qat.quantizer import QATQuantizer

    print("=" * 60)
    print("NVFP4 Offline Quantization — veRL API")
    print("=" * 60)
    print(f"Model:       {args.model_id}")
    print(f"Mode:        {args.mode}")
    print(f"Group size:  {args.group_size}")
    print(f"Device:      {args.device}")
    print(f"Output:      {args.output_dir}")
    print(f"veRL path:   {args.verl_path}")

    # ================================================================
    # Step 1: Load BF16 model
    # ================================================================
    print("\n[1/4] Loading BF16 model...")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded in {time.time() - t0:.0f}s — {total_params:.2f}B parameters")
    print(f"  Architecture: {config.architectures}")
    print(f"  Layers: {config.num_hidden_layers}")

    # ================================================================
    # Step 2: Quantize using QATQuantizer
    # ================================================================
    print(f"\n[2/4] Quantizing with QATQuantizer (mode={args.mode})...")
    t0 = time.time()

    quantizer = QATQuantizer(
        mode=args.mode,
        group_size=args.group_size,
        ignore_patterns=["lm_head", "embed_tokens", "re:.*mlp.gate$"],
        device=torch.device(args.device),
        param_dtype=torch.bfloat16,
    )

    # Collect named parameters as dict
    params_dict = dict(model.named_parameters())

    # quantize_with_fusion yields (name, quantized_tensor) pairs layer by layer
    new_state_dict = {}
    layer_count = 0
    last_layer = -1

    for name, tensor in quantizer.quantize_with_fusion(
        params_dict, target_device=torch.device("cpu")
    ):
        new_state_dict[name] = tensor

        # Progress: track layer transitions
        import re
        match = re.search(r"layers\.(\d+)\.", name)
        if match:
            cur_layer = int(match.group(1))
            if cur_layer != last_layer:
                last_layer = cur_layer
                layer_count += 1
                elapsed = time.time() - t0
                eta = (elapsed / max(layer_count, 1)) * (config.num_hidden_layers - layer_count)
                print(f"  [{layer_count}/{config.num_hidden_layers}] layer {cur_layer}  "
                      f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\n  Quantization done in {elapsed:.0f}s")
    print(f"  State dict entries: {len(new_state_dict)}")

    # ================================================================
    # Step 3: Save model
    # ================================================================
    print(f"\n[3/4] Saving quantized model...")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save quantized weights
    torch.save(new_state_dict, os.path.join(args.output_dir, "pytorch_model.bin"))
    print(f"  Saved pytorch_model.bin")

    # Save model config
    config.save_pretrained(args.output_dir)

    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)

    # Save quantization config for vLLM
    quant_config = build_quantization_config_json(args.group_size, args.mode)
    with open(os.path.join(args.output_dir, "quantization_config.json"), "w") as f:
        json.dump(quant_config, f, indent=2)
    print(f"  Saved quantization_config.json")

    # ================================================================
    # Step 4: Verify
    # ================================================================
    print(f"\n[4/4] Verification...")

    # Check key parameters exist
    param_names = list(new_state_dict.keys())
    weight_packed = [n for n in param_names if "weight_packed" in n]
    weight_scale = [n for n in param_names if "weight_scale" in n]
    weight_global = [n for n in param_names if "weight_global_scale" in n]

    print(f"  weight_packed entries:  {len(weight_packed)}")
    print(f"  weight_scale entries:   {len(weight_scale)}")
    print(f"  weight_global entries:  {len(weight_global)}")

    if len(weight_packed) == 0:
        print("\n  ⚠ WARNING: No weight_packed entries found!")
        print("  The model may not be properly quantized.")
        print("  Sample param names:")
        for n in param_names[:10]:
            print(f"    {n}")
        sys.exit(1)

    # Size comparison
    total_mb = sum(t.numel() * t.element_size() for t in new_state_dict.values()) / 1024 / 1024
    bf16_mb = total_params * 2 * 1024  # BF16 = 2 bytes × number of params in millions
    print(f"\n  Original BF16 size:   ~{total_params * 2:.0f} GB")
    print(f"  Quantized state dict:  {total_mb / 1024:.1f} GB")
    print(f"  Compression:           ~{total_params * 2 * 1024 / total_mb:.1f}x")

    # ================================================================
    # Done
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"✅ Done! Quantized model saved to: {args.output_dir}")
    print(f"{'=' * 60}")
    print(f"\nServe with vLLM on B300:")
    print(f"  CUDA_VISIBLE_DEVICES=2 vllm serve {os.path.abspath(args.output_dir)} \\")
    print(f"    --trust-remote-code \\")
    print(f"    --quantization compressed-tensors \\")
    print(f"    --dtype bfloat16 \\")
    print(f"    --port 8002 \\")
    print(f"    --gpu-memory-utilization 0.90 \\")
    print(f"    --max-model-len 4096")


if __name__ == "__main__":
    main()
