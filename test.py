from llama_cpp import Llama

llm = Llama(model_path="./models/qwen2.5-7b-instruct-q4_k_m.gguf")
output = llm("你好，请自我介绍一下。", max_tokens=100)
print(output["choices"][0]["text"])