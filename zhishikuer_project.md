# 智识库 (Zhishikuer) - 全栈本地私有化 AI 知识库项目指南

## 1. 项目概况
这是一个基于 Python 3.10 开发的私有化大模型知识库系统，旨在利用本地硬件资源（如您的 **AMD Ryzen 7 4800H / 16GB 内存**）实现高度隐私、无需联网的智能问答服务。

### 核心功能
* **RAG 检索增强生成**：基于本地文档（PDF/TXT）进行精准问答。
* **持久化数据库**：文档一次学习，永久保存（存储于 `./chroma_db`）。
* **隐私物理隔离**：内置“纯隐私”与“联网增强”双模式切换。
* **全栈集成**：FastAPI 后端 + Vue 3 前端，支持文档上传、列表管理与实时对话。

---

## 2. 目录结构
请确保您的项目目录如下：
```text
zhishikuer/
├── main.py                 # FastAPI 后端核心逻辑
├── models/
│   └── qwen2.5-7b-instruct-q4_k_m.gguf # 大模型文件
├── my_data/
│   └── info.txt            # 初始语料文档
├── static/                 # 静态资源文件夹
│   ├── index.html          # 前端界面文件
│   ├── vue.global.js       # Vue 3 离线库
│   ├── tailwind.min.js     # Tailwind CSS 离线库
│   └── font-awesome/       # 图标库文件夹
└── chroma_db/              # 自动生成的本地数据库文件夹
```

---

## 3. 后端代码 (main.py)
集成了最新的 **停止符 (Stop Sequences)** 和 **答案清洗 (Post-processing)** 逻辑，彻底解决复读指令和模拟对话问题。

```python
import os
import shutil
import sys
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- 配置参数 ---
MODEL_PATH = "./models/qwen2.5-7b-instruct-q4_k_m.gguf"
EMBEDDING_MODEL_PATH = r"D:\models\paraphrase-multilingual-MiniLM-L12-v2"
DB_DIR = "./chroma_db"

# --- 资源初始化 ---
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
if not os.path.exists(DB_DIR): os.makedirs(DB_DIR)
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8, 
    streaming=True,
    verbose=False,
    temperature=0.1,
    stop=["Assistant:", "用户:", "问：", "###", "Human:", "<|im_end|>"],
    repeat_penalty=1.2
)

search_tool = DuckDuckGoSearchRun()

system_prompt_str = (
    "你是一个专业的知识库助理。请直接根据【背景资料】回答用户问题。\n"
    "禁止输出任何引导性问题，禁止模拟对话。直接给出答案即可。\n\n"
    "【背景资料】：\n{context}"
)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt_str),
    ("human", "{input}"),
])

class ChatRequest(BaseModel):
    query: str
    use_online: bool = False

# --- 接口逻辑 ---

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    temp_path = f"./static/{file.filename}"
    with open(temp_path, "wb") as f: shutil.copyfileobj(file.file, f)
    try:
        loader = PyPDFLoader(temp_path) if temp_path.endswith(".pdf") else TextLoader(temp_path, encoding='utf-8')
        docs = loader.load()
        splits = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40).split_documents(docs)
        vector_db.add_documents(splits)
        return {"status": "success"}
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

@app.get("/files")
async def get_files():
    data = vector_db.get()
    sources = list(set([os.path.basename(m.get("source", "未知")) for m in data.get("metadatas", [])]))
    return {"files": sources}

@app.post("/chat")
async def chat(request: ChatRequest):
    combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
    qa_chain = create_retrieval_chain(vector_db.as_retriever(), combine_docs_chain)
    
    if not request.use_online:
        res = qa_chain.invoke({"input": request.query})
        full_answer = res["answer"]
        # 清洗逻辑：只保留 Assistant: 之后的内容，防止复读
        final_answer = full_answer.split("Assistant:")[-1].strip() if "Assistant:" in full_answer else full_answer.strip()
        return {"answer": final_answer, "mode": "本地库模式"}
    else:
        retrieval_tool = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever())
        tools = [
            Tool(name="本地库", func=retrieval_tool.run, description="查找私有信息"),
            Tool(name="联网搜索", func=search_tool.run, description="查找实时新闻")
        ]
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        res = agent.run(request.query)
        return {"answer": res, "mode": "联网增强模式"}

@app.delete("/clear")
async def clear():
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        os.makedirs(DB_DIR)
        global vector_db
        vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return {"status": "cleared"}

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 4. 前端界面 (static/index.html)
已适配离线资源引用及上传后自动刷新逻辑。

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>智识库 - 离线全栈版</title>
    <script src="/static/vue.global.js"></script>
    <script src="/static/tailwind.min.js"></script>
    <link rel="stylesheet" href="/static/font-awesome/css/all.min.css">
</head>
<body class="bg-gray-50 h-screen flex flex-col">
    <div id="app" class="flex flex-1 overflow-hidden">
        <!-- 侧边栏 -->
        <div class="w-64 bg-slate-900 text-white flex flex-col p-4">
            <h2 class="text-xl font-bold mb-8 flex items-center border-b border-slate-700 pb-4">智识库</h2>
            <div class="mb-6">
                <input type="file" @change="uploadFile" class="hidden" id="up">
                <label for="up" class="block w-full text-center bg-blue-600 py-3 rounded-xl cursor-pointer hover:bg-blue-700">学习新文档</label>
                <button @click="clearDB" class="w-full text-[10px] text-slate-600 mt-4 hover:text-red-400">清空数据库</button>
            </div>
            <div class="flex-1 overflow-y-auto">
                <p class="text-[10px] text-slate-500 mb-3 uppercase">已掌握文档 ({{files.length}})</p>
                <div v-for="f in files" class="text-xs p-3 bg-slate-800 rounded mb-2 truncate border border-slate-700">{{f}}</div>
            </div>
        </div>
        <!-- 主界面 -->
        <div class="flex-1 flex flex-col">
            <header class="h-14 bg-white border-b flex items-center justify-between px-6">
                <div class="flex items-center space-x-2 bg-gray-100 p-1 rounded-full">
                    <button @click="online = false" :class="!online ? 'bg-white shadow text-blue-600' : 'text-gray-400'" class="px-5 py-1 text-xs font-bold transition">🔒 纯本地</button>
                    <button @click="online = true" :class="online ? 'bg-white shadow text-green-600' : 'text-gray-400'" class="px-5 py-1 text-xs font-bold transition">🌐 联网</button>
                </div>
            </header>
            <div class="flex-1 overflow-y-auto p-6 space-y-4" id="chat">
                <div v-for="m in msgs" :class="m.role === 'user' ? 'justify-end' : 'justify-start'" class="flex">
                    <div :class="m.role === 'user' ? 'bg-blue-600 text-white' : 'bg-white border text-slate-800'" class="max-w-[75%] px-4 py-3 rounded-2xl shadow-sm text-sm">
                        {{m.text}}
                        <div v-if="m.mode" class="mt-1 text-[9px] opacity-30 text-right">{{m.mode}}</div>
                    </div>
                </div>
                <div v-if="loading" class="text-xs text-blue-400 animate-pulse">AI 正在深度思考...</div>
            </div>
            <div class="p-6 bg-white border-t">
                <div class="max-w-4xl mx-auto flex gap-4">
                    <input v-model="input" @keyup.enter="send" placeholder="输入问题..." class="flex-1 border rounded-2xl px-5 py-3 outline-none focus:border-blue-500">
                    <button @click="send" class="bg-blue-600 text-white px-10 rounded-2xl font-bold">发送</button>
                </div>
            </div>
        </div>
    </div>
    <script>
        const { createApp, ref, onMounted, nextTick } = Vue
        createApp({
            setup() {
                const msgs = ref([{ role: 'ai', text: '你好！请在左侧上传文档开始学习。' }])
                const input = ref(''), files = ref([]), online = ref(false), loading = ref(false)
                const refreshFiles = async () => {
                    const res = await fetch('/files'); const data = await res.json()
                    files.value = data.files
                }
                const send = async () => {
                    if(!input.value || loading.value) return
                    const q = input.value; msgs.value.push({ role: 'user', text: q })
                    input.value = '', loading.value = true
                    try {
                        const res = await fetch('/chat', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({ query: q, use_online: online.value })
                        })
                        const d = await res.json()
                        msgs.value.push({ role: 'ai', text: d.answer, mode: d.mode })
                    } finally {
                        loading.value = false
                        await nextTick(); document.getElementById('chat').scrollTop = 99999
                    }
                }
                const uploadFile = async (e) => {
                    const fd = new FormData(); fd.append('file', e.target.files[0])
                    loading.value = true
                    await fetch('/upload', { method: 'POST', body: fd })
                    await refreshFiles(); loading.value = false
                }
                const clearDB = async () => {
                    if(confirm('清空数据库吗？')) { await fetch('/clear', { method: 'DELETE' }); refreshFiles() }
                }
                onMounted(refreshFiles); return { msgs, input, files, online, loading, send, uploadFile, clearDB }
            }
        }).mount('#app')
    </script>
</body>
</html>
```

---

## 5. 硬件与运行建议
* **处理器**: Ryzen 7 4800H 已配置 `n_threads=8` 充分发挥 8 核 16 线程性能。
* **内存**: 16GB 建议在运行时关闭不必要的 Chrome 标签页。
* **脱网运行**: 确保 `static` 目录下的 Vue/Tailwind/Font-Awesome 已下载完毕。
