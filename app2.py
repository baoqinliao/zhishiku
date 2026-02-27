import os
# 1. 基础导入
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 2. 核心 LLM 导入 (确保你之前安装成功了 llama-cpp-python)
from langchain_community.llms import LlamaCpp

# 3. 问答链导入 (注意这里的路径变化)
from langchain.chains import RetrievalQA

# --- 配置区 ---
MODEL_PATH = "./models/qwen2.5-7b-instruct-q4_k_m.gguf" # 确认文件名正确
DOC_PATH = "./my_data/info.txt"  # 放一个包含你知识的txt文件

# 1. 加载文档并切分
print("正在读取知识库...")
# 如果是PDF就用PyPDFLoader
loader = TextLoader(DOC_PATH, encoding='utf-8')
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# 2. 文本向量化 (首次运行会下载一个约400MB的微型模型，用于理解语义)
print("正在将文字转为数字向量...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 3. 创建本地数据库
vector_db = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="./chroma_db")

# 4. 加载大模型 (针对你的 Ryzen 7 优化)
print("正在加载大模型到内存...")
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=2048,      # 上下文长度
    n_threads=8,     # 你的 CPU 是 8 核
    n_batch=512,     # 批处理大小
    verbose=False
)

# 5. 创建问答链
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever())

# 6. 交互
print("\n✅ 知识库加载完成！你可以开始提问了（输入 exit 退出）")
while True:
    query = input("\n用户: ")
    if query.lower() == 'exit': break
    print("AI 正在检索并思考...")
    res = qa.invoke(query)
    print(f"AI: {res['result']}")