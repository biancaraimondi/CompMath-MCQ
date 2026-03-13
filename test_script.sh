#!/bin/bash
models_name=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-Math-7B-Instruct"
  "Qwen/Qwen3-4B-Instruct-2507"
  "Qwen/Qwen3-30B-A3B-Instruct-2507"
  "Qwen/Qwen3-Coder-30B-A3B-Instruct"

  "google/gemini-3-flash-preview"
  "openai/gpt-5"
  "anthropic/claude-sonnet-4.5"
)

for model_name in "${models_name[@]}"; do
  echo "Evaluating model: $model_name"
  python test.py --model_name "$model_name"
done