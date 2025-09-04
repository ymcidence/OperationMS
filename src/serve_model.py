import vllm
from vllm import LLM, LLMPredictor

model_name = "Qwen/Qwen3-0.6B"
predictor = LLMPredictor(model_name=model_name)

server = LLM(predictor=predictor)
server.serve(host="0.0.0.0", port=8000)
