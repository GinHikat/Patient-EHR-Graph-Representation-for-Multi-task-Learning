docker run -it --rm \
  -p 8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --ipc=host \
  rocm/vllm:latest \
  python3 -m vllm.entrypoints.openai.api_server \
  --model google/medgemma-4b-it \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --port 8080
