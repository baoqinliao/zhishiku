# 🚀 智识库 (Zhishikuer) - Qwen2.5-14B GPU 加速部署指南

本指南专门针对 **NVIDIA RTX 4090 (24GB)** 硬件环境编写，旨在通过 **CUDA 12.2** 实现极速推理 [cite: a0c3130c-6220-49c9-b583-8e006e6a9167.jpg]。

---

## 🛠️ 第一步：下载模型文件 (GGUF 格式)

我们选择 **Q4_K_M** 量化版本，它在保持 14B 模型 99% 能力的同时，文件大小适中，显存占用仅约 9-10GB。

* **推荐下载地址 (Hugging Face)**:
  [Qwen2.5-14B-Instruct-GGUF (Bartowski 维护版)](https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf)
* **备用地址 (ModelScope 魔搭)**:
  在 ModelScope 搜索 `Qwen2.5-14B-Instruct-GGUF` 并下载。

**操作要求**：将下载好的文件放入项目的 `./models/` 文件夹下。

---

## 🛠️ 第二步：重构 GPU 环境

GPU 加速需要专门编译的 `llama-cpp-python` 库。

1. **卸载旧版本**：
   ```bash
   pip uninstall llama-cpp-python -y
   ```

2. **设置 CUDA 编译环境变量 (Windows PowerShell)**：
   ```powershell
   $env:CMAKE_ARGS="-DLLAMA_CUDA=on"
   $env:FORCE_CMAKE=1
   ```

3. **重新安装 GPU 版**：
   ```bash
   pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
   ```
   *注意：安装过程中如果出现 NVIDIA 相关报错，请确认您的显卡驱动版本为 535.171.04 或更高 [cite: a0c3130c-6220-49c9-b583-8e006e6a9167.jpg]。*

---

## 🛠️ 第三步：更新 main.py 核心参数

在您的 `main.py` 中，针对 14B 模型和 4090 显存进行如下优化：

```python
# 修改模型路径
MODEL_PATH = "./models/Qwen2.5-14B-Instruct-Q4_K_m.gguf"

# 重新初始化 LLM
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,    # 【关键】将 14B 模型的所有层全部搬进 4090 显存
    n_ctx=4096,         # 显存富余，可以调大上下文窗口
    n_batch=512,        # 增大批处理以加速
    n_threads=8,        # 保持 8 线程处理非推理逻辑
    streaming=True,
    verbose=False,
    temperature=0.1,
    stop=["Assistant:", "用户:", "问：", "###", "Human:", "<|im_end|>"],
    repeat_penalty=1.1
)

# Embedding 也迁移到显卡加速
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_PATH,
    model_kwargs={'device': 'cuda'}
)
```

---

## 🛠️ 第四步：清洗输出逻辑 (解决复读问题)

为了彻底解决您提到的“Assistant: 前多余回答”问题 [cite: image_1c26be.png]，请确保接口逻辑包含以下清洗步骤：

```python
@app.post("/chat")
async def chat(request: ChatRequest):
    # 执行检索问答
    res = qa_chain.invoke({"input": request.query})
    raw_answer = res["answer"]
    
    # 强制截断多余的引导词
    if "Assistant:" in raw_answer:
        final_answer = raw_answer.split("Assistant:")[-1].strip()
    else:
        final_answer = raw_answer.strip()
        
    return {"answer": final_answer, "mode": "GPU 加速模式"}
```

---

## 🛠️ 第五步：验证运行

1. **启动后端**：运行 `python main.py`。
2. **监控显存**：打开终端运行 `nvidia-smi`。您应该能看到显存占用增加了约 10GB 左右 [cite: a0c3130c-6220-49c9-b583-8e006e6a9167.jpg]。
3. **体验速度**：在前端提问，Qwen2.5-14B 的回答应该会瞬间喷涌而出。
