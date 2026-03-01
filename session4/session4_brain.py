from llama_cpp import Llama

# 1. 加载模型 (请确保你已下载 .gguf 文件并放在相应路径)
llm = Llama(
    model_path="../models/qwen2.5-7b-instruct-q4_k_m.gguf",
    n_ctx=2048,  # 上下文长度
    n_threads=4  # 根据你的 CPU 核心数调整
)


def generate_answer(query, context_list):
    """
    query: 用户的问题
    context_list: 从 ChromaDB 搜出来的文本片段
    """
    # 构造“喂”给 AI 的背景资料
    context_text = "\n".join([f"资料{i + 1}: {c}" for i, c in enumerate(context_list)])

    prompt = f"""你是一个私有知识库助手。请根据以下已知信息回答问题。
如果信息中没有提到，请直说“库中暂无相关记录”。

【已知信息】：
{context_text}

【用户问题】：
{query}

【正式回答】："""

    print("🤖 AI 正在深度思考...")
    response = llm(prompt, max_tokens=512, stop=["【", "\n\n"])
    return response["choices"][0]["text"]


# 测试一下：
if __name__ == "__main__":
    # 模拟从第三阶段搜出来的资料
    test_context = [
        "高性能服务器 SRV-2025 的单价为 25000 元。",
        "仓库位置在 A 区-01。"
    ]
    ans = generate_answer("服务器多少钱？在哪里？", test_context)
    print(f"\nAI 回复：{ans}")