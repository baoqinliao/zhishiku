import os
import shutil
import sys
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 核心 AI 库
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, RetrievalQA
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(title="智识库 Pro")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- 配置 (参考你的原始参数) ---
MODEL_PATH = "./models/qwen2.5-7b-instruct-q4_k_m.gguf"
EMBEDDING_MODEL_PATH = r"D:\models\paraphrase-multilingual-MiniLM-L12-v2"
DB_DIR = "./chroma_db"

# --- 初始化持久化资源 ---
print("--- 智识库系统启动中 ---")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
# 建立持久化向量数据库
if not os.path.exists(DB_DIR): os.makedirs(DB_DIR)
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# 加载 LlamaCpp
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,  # 匹配 4800H 核心
    streaming=True,
    verbose=False,
    temperature=0.1,
    stop=["<|im_start|>", "<|im_end|>", "用户:", "问：", "Human:", "###"],
    repeat_penalty=1.2
)

search_tool = DuckDuckGoSearchRun()

# --- 提示词定义 ---
system_prompt_str = (
    "你是一个基于本地资料的问答助手。你的任务是根据【背景资料】简洁回答问题。\n\n"
    "【背景资料】：\n{context}\n\n"
    "【要求】：直接给出答案，不要重复背景资料的标题。"
)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt_str),
    ("human", "{input}"),
])


class ChatRequest(BaseModel):
    query: str
    use_online: bool = False


# --- API 接口 ---

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """上传并自动学习文档"""
    temp_path = f"./static/{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # 支持 PDF 和 TXT
        loader = PyPDFLoader(temp_path) if temp_path.endswith(".pdf") else TextLoader(temp_path, encoding='utf-8')
        docs = loader.load()
        splits = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40).split_documents(docs)
        # 增量添加到本地数据库
        vector_db.add_documents(splits)
        return {"status": "success"}
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)


@app.get("/files")
async def get_files():
    """从数据库元数据中提取唯一文件名列表"""
    data = vector_db.get()
    if not data or not data["metadatas"]:
        return {"files": []}
    # 提取 source 字段并去重
    sources = list(set([os.path.basename(m.get("source", "未知文档")) for m in data["metadatas"]]))
    return {"files": sources}


@app.post("/chat")
async def chat(request: ChatRequest):
    """问答：隐私/联网双模式"""
    combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
    qa_chain = create_retrieval_chain(vector_db.as_retriever(), combine_docs_chain)

    if not request.use_online:
        res = qa_chain.invoke({"input": request.query})

        full_answer = res["answer"]

        # 【新增清洗逻辑】：只保留 "Assistant:" 之后的内容
        if "Assistant:" in full_answer:
            final_answer = full_answer.split("Assistant:")[-1].strip()
        else:
            final_answer = full_answer.strip()

        return {"answer": final_answer, "mode": "本地库模式"}


        #
        # return {"answer": res["answer"], "mode": "本地库模式"}
    else:
        retrieval_tool = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever())
        tools = [
            Tool(name="本地知识库", func=retrieval_tool.run, description="查找私有信息"),
            Tool(name="联网搜索", func=search_tool.run, description="查找实时新闻")
        ]
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        res = agent.run(request.query)
        return {"answer": res, "mode": "联网增强模式"}


@app.delete("/clear")
async def clear():
    """一键清空本地知识库"""
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        os.makedirs(DB_DIR)
        # 重新初始化全局对象
        global vector_db
        vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return {"status": "cleared"}


# 挂载本地静态资源文件夹
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)