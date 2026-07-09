#!/bin/bash
# launch_three_endpoints.sh
# 在 tmux 中启动三个 vLLM 端点：BF16 / FP8 / NVFP4
#
# 用法:
#   chmod +x launch_three_endpoints.sh
#   bash launch_three_endpoints.sh

MODEL_BF16=/workspace/volume/distributed-training-softdata/models/Qwen3-8B
MODEL_FP8=/workspace/volume/distributed-training-softdata/models/Qwen3-8B
MODEL_FP4=/workspace/volume/pengxiong/models/Qwen3-8B-NVFP4

SESSION=vllm-endpoints

# 创建 tmux session（如果已存在则复用）
tmux new-session -d -s $SESSION -n bf16

# ── Window 0: BF16 ──────────────────────────────────────
tmux send-keys -t $SESSION:bf16 "
  echo '=== BF16 — Port 8000 — GPU 0 ==='
  CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_BF16 \
    --trust-remote-code \
    --dtype bfloat16 \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    2>&1 | tee /tmp/vllm_bf16.log
" C-m

# ── Window 1: FP8 ───────────────────────────────────────
tmux new-window -t $SESSION -n fp8
tmux send-keys -t $SESSION:fp8 "
  echo '=== FP8 — Port 8001 — GPU 1 ==='
  CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL_FP8 \
    --trust-remote-code \
    --quantization fp8 \
    --port 8001 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    2>&1 | tee /tmp/vllm_fp8.log
" C-m

# ── Window 2: NVFP4 ─────────────────────────────────────
tmux new-window -t $SESSION -n nvfp4
tmux send-keys -t $SESSION:nvfp4 "
  echo '=== NVFP4 — Port 8002 — GPU 2 ==='
  CUDA_VISIBLE_DEVICES=2 vllm serve $MODEL_FP4 \
    --trust-remote-code \
    --quantization modelopt \
    --dtype bfloat16 \
    --port 8002 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    2>&1 | tee /tmp/vllm_nvfp4.log
" C-m

echo "Done! tmux session: $SESSION"
echo ""
echo "Attach:   tmux attach -t $SESSION"
echo "Detach:   Ctrl+B, D"
echo "Switch:   Ctrl+B, 0/1/2  or  Ctrl+B, n"
echo "List:     tmux ls"
echo "Kill:     tmux kill-session -t $SESSION"
echo ""
echo "Wait ~30s then check:"
echo "  curl http://localhost:8000/health"
echo "  curl http://localhost:8001/health"
echo "  curl http://localhost:8002/health"
