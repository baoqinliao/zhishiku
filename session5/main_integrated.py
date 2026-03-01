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






from fastapi import UploadFile, File, Form
import shutil


@app.post("/chat_vision")
async def chat_vision(
        prompt: str = Form(...),
        file: UploadFile = File(None)  # 支持可选的图片上传
):
    if file:
        # 1. 保存用户上传的临时图片
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. 调用 VLM 模型进行推理
        # 这里会调用类似 llm.create_chat_completion(messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": temp_path}]}] )
        answer = vision_engine.ask_about_image(temp_path, prompt)
        return {"answer": answer, "image_url": f"/temp/{file.filename}"}
    else:
        # 回到之前的 RAG 文本逻辑
        return integrated_ask(prompt)







if __name__ == "__main__":
    # 启动命令
    uvicorn.run(app, host="127.0.0.1", port=8000)