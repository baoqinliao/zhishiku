import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

# 导入你之前写的 RAG 引擎
from rag_engine import integrated_ask

app = FastAPI()

# 1. 挂载图片文件夹（让前端能看到 session2 里的图片）
# 注意修改这里的路径，确保指向你存放 11.jpg 的地方
# 修改这一行，让它指向你 session4 文件夹下的 images 目录
app.mount("/images", StaticFiles(directory="images"), name="images")

# 2. 设置模板文件夹
templates = Jinja2Templates(directory="templates")


class QueryModel(BaseModel):
    prompt: str


# --- 路由：主页面 ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# --- 路由：对话 API ---
@app.post("/chat")
async def chat(query_data: QueryModel):
    # 调用 RAG 引擎
    result = integrated_ask(query_data.prompt)


    print(result)

    # 路径转换：将本地绝对路径转为 Web 可访问路径
    web_sources = []
    for src in result["sources"]:
        filename = os.path.basename(src)
        web_sources.append(f"/images/{filename}")

    return {
        "answer": result["answer"],
        "sources": web_sources
    }


if __name__ == "__main__":
    # 启动命令
    uvicorn.run(app, host="127.0.0.1", port=8000)