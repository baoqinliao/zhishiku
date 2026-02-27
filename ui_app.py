import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 页面配置 ---
st.set_page_config(page_title="我的私有智识库", page_icon="🤖")
st.title("🤖 我的本地 AI 知识库")
st.caption("基于 AMD Ryzen 7 4800H + 16GB 内存驱动")  #

# --- 常量配置 ---
MODEL_PATH = "./models/qwen2.5-7b-instruct-q4_k_m.gguf"
DOC_PATH = "./my_data/info.txt"
EMBEDDING_PATH = r"D:\models\paraphrase-multilingual-MiniLM-L12-v2"


# --- 核心逻辑缓存 (避免重复加载模型) ---
@st.cache_resource
def init_rag():
    # 1. 加载文档
    loader = TextLoader(DOC_PATH, encoding='utf-8')
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    splits = text_splitter.split_documents(docs)

    # 2. 向量数据库
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_PATH)
    vector_db = Chroma.from_documents(documents=splits, embedding=embeddings)

    # 3. 加载大模型 (针对 8 核 CPU 优化)
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=8,  # 充分利用处理器核心
        verbose=False,
        temperature=0.1,
        stop=["用户:", "问：", "【", "Human:"]
    )

    # 4. 组装链
    system_prompt = "你是一个助手，请根据背景资料回答问题：\n\n{context}"
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    combine_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vector_db.as_retriever(), combine_chain)


# 初始化 RAG
if os.path.exists(MODEL_PATH) and os.path.exists(DOC_PATH):
    qa_chain = init_rag()
else:
    st.error("❌ 找不到模型或 info.txt，请检查路径！")
    st.stop()

# --- 聊天界面 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt_input := st.chat_input("问我任何关于知识库的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # 生成 AI 回答
    with st.chat_message("assistant"):
        with st.spinner("AI 正在翻阅文档并思考..."):
            res = qa_chain.invoke({"input": prompt_input})
            answer = res["answer"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})