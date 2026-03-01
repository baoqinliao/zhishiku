import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag_engine import integrated_ask  # 导入你之前的引擎

app = FastAPI()

# 1. 解决跨域问题：允许 Vue (通常在 5173 端口) 访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议指定具体域名
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 静态文件挂载：让浏览器能通过 URL 访问你的 11.jpg
# 假设你的图片在 D:\rag\session2 里
app.mount("/images", StaticFiles(directory="../session2"), name="images")


class QueryModel(BaseModel):
    prompt: str


@app.post("/chat")
async def chat(query_data: QueryModel):
    try:
        result = integrated_ask(query_data.prompt)

        # 修正图片路径：将本地绝对路径转换为浏览器可访问的 URL
        web_sources = []
        for src in result["sources"]:
            filename = os.path.basename(src)
            web_sources.append(f"http://localhost:8000/images/{filename}")

        return {
            "answer": result["answer"],
            "sources": web_sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)