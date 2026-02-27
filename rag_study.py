import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import ollama

# 1. 加载本地文档 (假设当前目录下有个 data 文件夹)
if not os.path.exists('data'):
    os.makedirs('data')
    with open('data/test.txt', 'w', encoding='utf-8') as f:
        f.write("Gemini 是一款强大的 AI 助手，它能帮你写代码。")

loader = DirectoryLoader('./data', glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

# 2. 文档切分 (为了让 CPU 处理更高效)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

# 3. 向量化 (使用轻量级 CPU 嵌入模型)
# 这个模型会自动下载，仅几十MB，在 4800H 上运行飞快
embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese",cache_folder="D:/LocalAI/models")
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)


# 4. 问答函数
def local_knowledge_chat(query):
    # A. 检索：从向量库找最相关的 3 片内容
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # B. 生成：调用 Ollama 里的模型
    prompt = f"你是一个助手。请基于以下已知内容回答问题：\n内容：{context}\n问题：{query}"

    response = ollama.chat(model='qwen2.5:7b', messages=[
        {'role': 'user', 'content': prompt}
    ])
    return response['message']['content']


# 5. 测试提问
print("本地知识库已就绪！")
print(local_knowledge_chat("Gemini 是什么？"))