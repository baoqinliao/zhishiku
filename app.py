import os
import sys
import time

# --- 环境检查 ---
print(f"--- 启动中 ---\n当前 Python: {sys.executable}")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.llms import LlamaCpp
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.callbacks import StreamingStdOutCallbackHandler

    print("✅ 核心库导入成功！")
except ImportError as e:
    print(f"❌ 导入失败: {e}");
    sys.exit()

# --- 配置 ---
MODEL_PATH = "./models/qwen2.5-7b-instruct-q4_k_m.gguf"
DOC_PATH = "./my_data/info.txt"
EMBEDDING_MODEL_PATH = r"D:\models\paraphrase-multilingual-MiniLM-L12-v2"

# 1. 文档预处理
loader = TextLoader(DOC_PATH, encoding='utf-8')
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
splits = text_splitter.split_documents(docs)

# 2. 向量库加载
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
vector_db = Chroma.from_documents(documents=splits, embedding=embeddings)

# 3. 加载大模型（核心优化点）
print(f"正在加载模型...")
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    verbose=False,
    temperature=0.3,  # 稍微给点随机性，防止它卡死
    # 增加物理停止符，只要看到这些，程序就强行掐断输出，防止它“演戏”
    stop=["<|im_start|>", "<|im_end|>", "用户:", "问：", "Human:", "###"],
    repeat_penalty=1.1  # 惩罚重复，防止它复读
)

# 4. 组装问答链（提示词结构优化）
# 我们使用更干净的分隔符，防止模型把指令当成要回答的内容
system_prompt = (
    "你是一个基于本地资料的问答助手。你的任务是根据【背景资料】简洁回答问题。\n\n"
    "【背景资料】：\n{context}\n\n"
    "【要求】：直接给出答案，不要重复背景资料的标题，不要输出无关的符号。"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(vector_db.as_retriever(), combine_docs_chain)

# 5. 循环对话
print("\n" + "=" * 30)
print("🎉 智识库已准备就绪！输入 exit 退出。")
print("=" * 30)

while True:
    user_input = input("\n问：")
    if user_input.lower() in ['exit', 'quit']: break

    print("🧠 正在组织语言...", end="\r", flush=True)

    try:
        # invoke 会直接通过 StreamingStdOutCallbackHandler 把字蹦出来
        # 我们用一个变量接收，但不打印它，防止重复
        _ = qa_chain.invoke({"input": user_input})
        print("\n" + "=" * 30)
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")