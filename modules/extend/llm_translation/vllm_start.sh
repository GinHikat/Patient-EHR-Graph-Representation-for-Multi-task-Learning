docker run -it --rm \
  -p 8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --ipc=host \
  rocm/vllm:latest \
  python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-72B-Instruct \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --port 8080
