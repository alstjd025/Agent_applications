# bash run_one_server.sh
# # 또는
# TENSOR_PARALLEL_SIZE=1 bash run_one_server.sh
# ```

# **nvidia-smi 결과:**
# ```
# GPU 0: 42GB 사용
# GPU 1: 거의 사용 안 함

# TENSOR_PARALLEL_SIZE=2 bash run_one_server.sh
# GPU 2개 사용



#!/usr/bin/env bash
set -euo pipefail

# ✅ vLLM 설치된 venv를 강제로 사용
source /workspace/vllm/ms_local/scripts/00_env.sh
source "$VENV_DIR/bin/activate"

LOG_DIR="/workspace/vllm/ms_local/agent_single_model/logs"
mkdir -p "$LOG_DIR"
PORT="${PORT:-8001}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"

# 기존 서버 정리
pkill -9 -f "vllm.entrypoints.openai.api_server" || true

echo "[*] Starting vLLM server on :$PORT"
echo "[*] Model: $MODEL"
echo "[*] Max model length: 8192"
echo "[*] GPU memory utilization: 0.90"
echo "[*] Tensor parallel size: ${TENSOR_PARALLEL_SIZE:-1}"
echo "[*] Logs will be saved to $LOG_DIR/server.log"

# Tensor parallel size 설정 (기본값: 1 = 단일 GPU)
# 2개 GPU 사용시: TENSOR_PARALLEL_SIZE=2 bash run_one_server.sh
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --enable-log-requests \
  2>&1 | tee "$LOG_DIR/server.log"