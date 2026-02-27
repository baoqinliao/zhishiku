from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 1. 配置本地大模型 (替换成你下载的文件路径)
print("正在加载模型到 CPU，这可能需要几十秒...")
llm = LlamaCpp(
    model_path="./models/qwen2-7b-instruct-q4_k_m.gguf",
    n_ctx=2048,  # 上下文窗口
    n_threads=8,  # 你的 CPU 是 8 核，这里填 8 效率最高
    verbose=False
)

# 2. 加载并切分你的知识库文档 (以 PDF 为例)
print("正在处理文档...")
loader = PyPDFLoader("你的文档.pdf")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_splits = text_splitter.split_documents(data)

# 3. 文本向量化 (完全本地运行)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 4. 创建本地数据库
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

# 5. 构建问答链
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# 6. 提问
print("\n--- 知识库准备就绪 ---")
while True:
    question = input("\n问：")
    if question == 'exit': break

    print("思考中...")
    response = qa_chain.invoke(question)
    print(f"答：{response['result']}")