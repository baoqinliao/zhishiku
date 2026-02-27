# 📚 智识库 Python 3.10 + RTX 4090 极速部署手册

## 1. 系统要求
* **Python 版本**: 3.10.x
* **显卡**: NVIDIA RTX 4090 (24GB VRAM) [cite: a0c3130c-6220-49c9-b583-8e006e6a9167.jpg]
* **CUDA**: 12.2 [cite: a0c3130c-6220-49c9-b583-8e006e6a9167.jpg]

## 2. 环境安装指令
请按顺序在终端执行：
1. `pip install fastapi==0.109.2 uvicorn==0.27.1 python-multipart==0.0.9 langchain==0.1.9 langchain-community==0.0.24 langchain-core==0.1.26 langchain-huggingface==0.0.1 chromadb==0.4.24 sentence-transformers==2.5.1 pypdf==4.1.0 unstructured==0.12.5 duckduckgo-search==5.0`
2. `$env:CMAKE_ARGS="-DLLAMA_CUDA=on"`
3. `$env:FORCE_CMAKE=1`
4. `pip install llama-cpp-python==0.2.44 --force-reinstall --upgrade --no-cache-dir`

## 3. 模型下载
推荐 Qwen2.5-14B GGUF 版本，显存占用约为 10GB，剩余显存可保障系统流畅。

## 4. 常见坑点提醒
* **显存溢出**: 4090 显存非常充裕，如果遇到 OOM，请检查是否同时开启了其他大型 3D 程序 [cite: a0c3130c-6220-49c9-b583-8e006e6a9167.jpg]。
* **环境变量**: 确保 CUDA_PATH 环境变量已正确指向 12.2 版本安装路径。
