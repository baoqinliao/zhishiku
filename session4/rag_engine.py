import os
import chromadb
from llama_cpp import Llama
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

# 1. 启动大脑 (LLM)
llm = Llama(
    model_path="../models/qwen2.5-7b-instruct-q4_k_m.gguf",
    n_ctx=4096,  # 调大一点，方便读取更多背景
    n_threads=4
)

# 2. 启动记忆 (ChromaDB)
# 注意：这里需要和你第三阶段的模型参数保持完全一致
multimodal_ef = OpenCLIPEmbeddingFunction(
    model_name="ViT-B-32-quickgelu",
    checkpoint="laion400m_e32"
)
client = chromadb.PersistentClient(path="./my_multimodal_db")
collection = client.get_or_create_collection(
    name="multimodal_knowledge",
    embedding_function=multimodal_ef
)


def integrated_ask(query):
    # --- 步骤 A: 检索 ---
    print(f"🔍 正在从知识库寻找线索...")
    results = collection.query(query_texts=[query], n_results=10, # 搜 5 个，增加图片入选的概率
        include=['documents', 'metadatas', 'uris', 'distances'])

    print(results)

    # 提取文本片段和图片路径


    context_list = []
    source_urls = set()  # 使用集合去重

    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]

        # 1. 如果搜到的是直接的图片条目
        if meta.get('type') == 'image':
            # 这里的 uri 是我们在 A 段存入的绝对路径，转换成 web 路径
            filename = os.path.basename(results['uris'][0][i])
            source_urls.add(f"/images/{filename}")

        # 2. 如果搜到的是文本条目
        elif meta.get('type') == 'text':
            context_list.append(results['documents'][0][i])
            # 看看这个文本片段是不是来自于某个图片 (如 11.jpg)
            img_url = meta.get('img_url')
            if img_url:
                source_urls.add(img_url)
    # --- 步骤 B: 生成 ---

    context_text = "\n".join(context_list)
    prompt = f"""<|im_start|>system
你是一个专业的私有知识库助手。请根据已知信息，用简洁、专业的口吻回答用户问题。
如果信息不足，请委婉告知。<|im_end|>
<|im_start|>user
已知信息：
{context_text}

问题：{query}<|im_end|>
<|im_start|>assistant
"""

    print("🤖 AI 正在整合答案...")
    response = llm(prompt, max_tokens=512, stop=["<|im_end|>"])
    answer = response["choices"][0]["text"]

    return {
        "answer": answer,
        "sources": list(source_urls)
    }


if __name__ == "__main__":
    res = integrated_ask("那个绿色的护发素里含燕麦吗？多少钱？")
    print(f"\n【最终回答】:\n{res['answer']}")
    print(f"【参考图片】: {res['sources']}")