#!/bin/bash
# launch_three_endpoints.sh
# 在 tmux 中启动三个 vLLM 端点：BF16 / FP8 / NVFP4
#
# 用法:
#   bash launch_three_endpoints.sh [GPU_BF16] [GPU_FP8] [GPU_FP4]
#
# 示例:
#   bash launch_three_endpoints.sh 0 1 2        # 默认：各占一张卡
#   bash launch_three_endpoints.sh 0 0 1        # BF16+FP8 共享 GPU0, FP4 用 GPU1

# ── GPU 配置 ──────────────────────────────────────────────
GPU_BF16=${1:-5}
GPU_FP8=${2:-6}
GPU_FP4=${3:-7}

# ── 模型路径 ──────────────────────────────────────────────
MODEL_BF16=/workspace/volume/distributed-training-softdata/models/Qwen3-8B
MODEL_FP8=/workspace/volume/distributed-training-softdata/models/Qwen3-8B
MODEL_FP4=/workspace/volume/pengxiong/models/Qwen3-8B-NVFP4

# ── 端口 ──────────────────────────────────────────────────
PORT_BF16=8002
PORT_FP8=8003
PORT_FP4=8004

# ── vLLM 通用参数 ─────────────────────────────────────────
MEM_UTIL=0.90
MAX_LEN=4096

# ── 日志 ──────────────────────────────────────────────────
LOG_DIR=/tmp/vllm_logs
mkdir -p $LOG_DIR

SESSION=vllm-endpoints

# ── 如果 session 已存在，先杀掉 ───────────────────────────
tmux has-session -t $SESSION 2>/dev/null && tmux kill-session -t $SESSION

# ── 创建 session ──────────────────────────────────────────
tmux new-session -d -s $SESSION -n bf16

# ── Window 0: BF16 ───────────────────────────────────────
tmux send-keys -t $SESSION:bf16 "
  echo '=== BF16 | GPU ${GPU_BF16} | Port ${PORT_BF16} ==='
  CUDA_VISIBLE_DEVICES=${GPU_BF16} vllm serve ${MODEL_BF16} \
    --trust-remote-code \
    --dtype bfloat16 \
    --port ${PORT_BF16} \
    --gpu-memory-utilization ${MEM_UTIL} \
    --max-model-len ${MAX_LEN} \
    2>&1 | tee ${LOG_DIR}/bf16.log
" C-m

# ── Window 1: FP8 ────────────────────────────────────────
tmux new-window -t $SESSION -n fp8
tmux send-keys -t $SESSION:fp8 "
  echo '=== FP8  | GPU ${GPU_FP8} | Port ${PORT_FP8} ==='
  CUDA_VISIBLE_DEVICES=${GPU_FP8} vllm serve ${MODEL_FP8} \
    --trust-remote-code \
    --quantization fp8 \
    --port ${PORT_FP8} \
    --gpu-memory-utilization ${MEM_UTIL} \
    --max-model-len ${MAX_LEN} \
    2>&1 | tee ${LOG_DIR}/fp8.log
" C-m

# ── Window 2: NVFP4 ──────────────────────────────────────
tmux new-window -t $SESSION -n nvfp4
tmux send-keys -t $SESSION:nvfp4 "
  echo '=== NVFP4 | GPU ${GPU_FP4} | Port ${PORT_FP4} ==='
  CUDA_VISIBLE_DEVICES=${GPU_FP4} vllm serve ${MODEL_FP4} \
    --trust-remote-code \
    --quantization modelopt \
    --dtype bfloat16 \
    --port ${PORT_FP4} \
    --gpu-memory-utilization ${MEM_UTIL} \
    --max-model-len ${MAX_LEN} \
    2>&1 | tee ${LOG_DIR}/nvfp4.log
" C-m

# ── Done ──────────────────────────────────────────────────
echo "=== vLLM Endpoints ==="
echo "  BF16:  GPU ${GPU_BF16} → port ${PORT_BF16} | log: ${LOG_DIR}/bf16.log"
echo "  FP8:   GPU ${GPU_FP8} → port ${PORT_FP8}  | log: ${LOG_DIR}/fp8.log"
echo "  NVFP4: GPU ${GPU_FP4} → port ${PORT_FP4}  | log: ${LOG_DIR}/nvfp4.log"
echo ""
echo "tmux:  tmux attach -t ${SESSION}"
echo "check: curl http://localhost:${PORT_BF16}/health  # etc"
echo "kill:  tmux kill-session -t ${SESSION}"
