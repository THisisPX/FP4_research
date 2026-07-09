"""
Offline NVFP4 W4A16 quantization for Qwen3-8B on B300.

Produces a checkpoint that vLLM can serve with:
    vllm serve /path/to/Qwen3-8B-NVFP4 \\
        --trust-remote-code --quantization compressed-tensors --port 8002

Requirements:
    pip install compressed-tensors transformers torch accelerate
"""

import argparse
import json
import os
import shutil
import time

import torch
from transformers import AutoModelForCausalLM, AutoConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=str, default="Qwen/Qwen3-8B",
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./Qwen3-8B-NVFP4",
        help="Output directory for quantized model"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device for quantization (use CPU if OOM)"
    )
    parser.add_argument(
        "--group_size", type=int, default=16,
        help="NVFP4 block size (16 is standard for Blackwell)"
    )
    return parser.parse_args()


def build_compressed_tensors_config(hf_config, group_size):
    """Create the quantization config that vLLM's compressed-tensors
    backend expects."""
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "global_compression_ratio": 3.92,  # FP16 -> NVFP4 ≈ 4x
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "float",       # NVFP4 = E2M1 float format
                        "strategy": "group",    # Per-group quantization
                        "group_size": group_size,
                        "symmetric": True,
                        "actorder": False,
                        "scale_dtype": "float8_e4m3fn",  # FP8 scales
                    },
                    "input_activations": None,  # W4A16: activations stay BF16
                }
            },
        },
    }


def quantize_linear_layer(weight, group_size, device):
    """
    Quantize a single Linear weight matrix to NVFP4.
    Returns (packed_weight, weight_scale, weight_global_scale).

    NVFP4 format: E2M1 (1 sign, 2 exponent, 1 mantissa)
    Each 16-element group has one FP8 block scale + one FP32 global scale.
    """
    import numpy as np
    out_features, in_features = weight.shape
    assert in_features % group_size == 0, \
        f"in_features {in_features} must be divisible by group_size {group_size}"
    assert out_features % group_size == 0, \
        f"out_features {out_features} must be divisible by group_size {group_size}"

    num_groups_per_row = in_features // group_size  # K direction groups

    weight_f32 = weight.float().to(device)

    # Step 1: Global scale (FP32, one scalar per weight matrix row-direction groups)
    # For NVFP4, we need one global scale per (group_size, group_size) tile
    # global_scale shape: (out_features, num_groups_per_row)

    global_scale = torch.zeros(
        out_features, num_groups_per_row,
        dtype=torch.float32, device=device
    )

    for i in range(out_features):
        for j in range(num_groups_per_row):
            start_col = j * group_size
            end_col = start_col + group_size
            group_values = weight_f32[i, start_col:end_col]
            amax = torch.max(torch.abs(group_values))
            global_scale[i, j] = amax / 6.0  # E2M1 max representable value = 6.0

    # Step 2: Block-wise scale (FP8 E4M3, one per block)
    # Each (group_size, group_size) block has an FP8 scale
    num_blocks_row = out_features // group_size
    num_blocks_col = num_groups_per_row

    weight_scale = torch.zeros(
        num_blocks_row, num_blocks_col,
        dtype=torch.float8_e4m3fn, device=device
    )

    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            r_start = i * group_size
            r_end = r_start + group_size
            c_start = j * group_size
            c_end = c_start + group_size

            block_values = weight_f32[r_start:r_end, c_start:c_end]
            block_global_scale_min = torch.min(global_scale[r_start:r_end, c_start:c_end])

            # Local scale within block
            block_amax = torch.max(torch.abs(block_values))
            local_scale = block_amax / 6.0

            # Combined scale = global * local, clamped to FP8 E4M3 range
            combined = torch.clamp(
                block_global_scale_min * local_scale,
                min=1e-38, max=448.0  # FP8 E4M3 range
            )
            weight_scale[i, j] = combined.to(torch.float8_e4m3fn)

    # Step 3: Quantize weights to UINT8 packed format (2 x 4-bit values per byte)
    # E2M1 quantization grid: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
    E2M1_VALUES = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32, device=device
    )

    weight_packed = torch.zeros(
        out_features, in_features // 2,
        dtype=torch.uint8, device=device
    )

    for i in range(out_features):
        for j in range(0, in_features, 2):
            w0 = weight_f32[i, j]
            w1 = weight_f32[i, j + 1]

            # Find nearest E2M1 value for each weight
            # Use the global_scale at this position
            col_group = j // group_size
            col_group_2 = (j + 1) // group_size
            gs0 = global_scale[i, col_group]
            gs1 = global_scale[i, col_group_2]

            if gs0 > 0:
                nw0 = w0 / gs0
                best0 = torch.argmin(torch.abs(E2M1_VALUES - nw0))
            else:
                best0 = torch.tensor(0, device=device)

            if gs1 > 0:
                nw1 = w1 / gs1
                best1 = torch.argmin(torch.abs(E2M1_VALUES - nw1))
            else:
                best1 = torch.tensor(0, device=device)

            # Pack two 4-bit values into one uint8
            packed = (best0 & 0x0F) | ((best1 & 0x0F) << 4)
            weight_packed[i, j // 2] = packed.to(torch.uint8)

    return weight_packed.cpu(), weight_scale.cpu(), global_scale.cpu()


def main():
    args = parse_args()

    print(f"Loading model: {args.model_id}")
    print(f"Device: {args.device}")
    print(f"Group size: {args.group_size}")
    print(f"Output: {args.output_dir}")

    # ==============================================================
    # Step 1: Load model in BF16
    # ==============================================================
    print("\n[1/5] Loading model...")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)

    print(f"  Model loaded in {time.time() - t0:.0f}s")
    print(f"  Architecture: {config.architectures}")
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Parameters: {total_params:.2f}B")

    # ==============================================================
    # Step 2: Identify linear layers to quantize
    # ==============================================================
    print("\n[2/5] Identifying layers to quantize...")

    skip_patterns = [
        "lm_head",
        "model.embed_tokens",
    ]
    # Also skip layers with gate in name (MoE routers)

    quantizable_layers = []
    skipped_layers = []

    for name, param in model.named_parameters():
        if not name.endswith(".weight"):
            continue
        if param.dim() != 2:
            continue  # Not a Linear layer weight

        should_skip = False
        for pattern in skip_patterns:
            if pattern in name:
                should_skip = True
                break
        # Skip router/gate layers
        if "mlp.gate" in name or "gate_proj" in name:
            # gate_proj in Qwen3 is actually part of the MLP, not a router
            # We quantize it
            pass

        if should_skip:
            skipped_layers.append(name)
        elif param.shape[0] % args.group_size == 0 and param.shape[1] % args.group_size == 0:
            quantizable_layers.append((name, param))
        else:
            skipped_layers.append(name)

    print(f"  Quantizable layers: {len(quantizable_layers)}")
    print(f"  Skipped layers: {len(skipped_layers)}")
    for s in skipped_layers[:5]:
        print(f"    skip: {s}")
    if len(skipped_layers) > 5:
        print(f"    ... and {len(skipped_layers) - 5} more")

    # ==============================================================
    # Step 3: Quantize each layer
    # ==============================================================
    print("\n[3/5] Quantizing weights to NVFP4...")
    t0 = time.time()

    quantized_state_dict = {}
    total_original_bytes = 0
    total_quantized_bytes = 0

    # Calculate total parameters for progress
    total_layers = len(quantizable_layers)

    for idx, (name, param) in enumerate(quantizable_layers):
        weight_before_path = name.replace(".weight", ".weight_packed")
        scale_path = name.replace(".weight", ".weight_scale")
        global_scale_path = name.replace(".weight", ".weight_global_scale")

        # Keep BF16 copy for non-quantized layers
        original_param = param.data.clone()

        # Quantize
        try:
            packed, scale, global_scale = quantize_linear_layer(
                original_param, args.group_size, args.device
            )

            quantized_state_dict[weight_before_path] = packed
            quantized_state_dict[scale_path] = scale
            quantized_state_dict[global_scale_path] = global_scale

            total_original_bytes += original_param.numel() * 2  # BF16 = 2 bytes
            total_quantized_bytes += packed.numel() * 1 + scale.numel() * 1 + global_scale.numel() * 4
        except Exception as e:
            print(f"  ERROR quantizing {name}: {e}")
            quantized_state_dict[name] = original_param.cpu()
            skip_patterns.append(name)

        if (idx + 1) % 50 == 0 or idx == total_layers - 1:
            elapsed = time.time() - t0
            eta = (elapsed / (idx + 1)) * (total_layers - idx - 1)
            print(f"  [{idx + 1}/{total_layers}] "
                  f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s   "
                  f"current: {name}")

    print(f"\n  Quantization complete in {time.time() - t0:.0f}s")
    compression_ratio = total_original_bytes / total_quantized_bytes if total_quantized_bytes > 0 else 1
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Original size: {total_original_bytes / 1e9:.2f} GB")
    print(f"  Quantized size: {total_quantized_bytes / 1e9:.2f} GB")

    # ==============================================================
    # Step 4: Add unmodified weights
    # ==============================================================
    print("\n[4/5] Collecting unmodified weights...")
    for name, param in model.named_parameters():
        if name in quantized_state_dict:
            continue
        # Check if this param was already handled by the quantizer
        base_name = name.replace(".weight", "")
        if any(base_name in k for k in quantized_state_dict):
            continue
        quantized_state_dict[name] = param.data.cpu()

    # Also save non-parameter buffers
    for name, buffer in model.named_buffers():
        quantized_state_dict[name] = buffer.cpu()

    print(f"  Total state dict entries: {len(quantized_state_dict)}")

    # ==============================================================
    # Step 5: Save model
    # ==============================================================
    print(f"\n[5/5] Saving quantized model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Save quantized weights
    torch.save(quantized_state_dict, os.path.join(args.output_dir, "pytorch_model.bin"))

    # Save quantization config that vLLM recognizes
    quant_config = build_compressed_tensors_config(config, args.group_size)
    with open(os.path.join(args.output_dir, "quantization_config.json"), "w") as f:
        json.dump(quant_config, f, indent=2)

    # Copy config files from original model
    source_dir = args.model_id if os.path.isdir(args.model_id) else None
    if source_dir is None:
        # HuggingFace cached model
        from transformers.utils import WEIGHTS_NAME
        source_dir = args.model_id

    # Save config to output
    config.save_pretrained(args.output_dir)

    # Copy tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    # Save model metadata for compressed-tensors compatibility
    quant_meta = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_config": quant_config["quantization_config"],
        "ignore": skip_patterns[:20],  # first 20 skipped layers for reference
    }
    with open(os.path.join(args.output_dir, "quantize_config.json"), "w") as f:
        json.dump(quant_meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! Quantized model saved to: {args.output_dir}")
    print(f"{'='*60}")
    print(f"\nTo serve with vLLM on B300:")
    print(f"  vllm serve {os.path.abspath(args.output_dir)} \\")
    print(f"    --trust-remote-code \\")
    print(f"    --quantization compressed-tensors \\")
    print(f"    --dtype bfloat16 \\")
    print(f"    --port 8002 \\")
    print(f"    --gpu-memory-utilization 0.90 \\")
    print(f"    --max-model-len 4096")


if __name__ == "__main__":
    main()
