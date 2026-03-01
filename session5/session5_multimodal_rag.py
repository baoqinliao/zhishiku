import os
import re

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 1. 初始化多模态 Embedding 模型 (使用公开的 CLIP 模型)
# 这个模型能同时理解图片和中文/英文
model_name = "ViT-B-32-quickgelu"
checkpoint = "laion400m_e32"
multimodal_ef = OpenCLIPEmbeddingFunction(model_name=model_name, checkpoint=checkpoint)
image_loader = ImageLoader()

# 2. 初始化 ChromaDB
client = chromadb.PersistentClient(path="./my_multimodal_db")
collection = client.get_or_create_collection(
    name="multimodal_knowledge",
    embedding_function=multimodal_ef,
    data_loader=image_loader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

import numpy as np


def ingest_multimodal_data(folder_path):
    # --- A. 图片入库 (视觉特征) ---
    for item in os.listdir(folder_path):
        if item.lower().endswith(('.jpg', '.png', '.jpeg')):
            file_path = os.path.join(folder_path, item)

            try:
                # 1. 修复点：加载图像数据
                # 必须将图片打开并转为 numpy 数组，传给 images 参数
                img = Image.open(file_path).convert("RGB")
                img_array = np.array(img)

                print(f"🖼️ 正在编码图片视觉特征: {item},{file_path}")
                collection.upsert(
                    ids=[f"img_{item}"],
                    images=[img_array],  # 关键修复：提供图像数据
                    uris=[file_path],  # 依然保存路径以便后续展示
                    metadatas=[{"type": "image", "source": item}]
                )
            except Exception as e:
                print(f"❌ 图像 {item} 加载失败: {e}")

    # --- B. 文本入库 (语义特征) ---
    kb_path = os.path.join(folder_path, "final_knowledge_base.txt")
    if os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as f:
            content = f.read()

        sections = re.split(r'={10,}\n📥 文件名: (.*?)\n={10,}', content)

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        # chunks = splitter.split_text(content)

        for i in range(1, len(sections), 2):
            source_name = sections[i].strip()
            content = sections[i + 1].strip()

            # 为该文件下的所有切片生成 web 可访问的图片 URL (如果是图片的话)
            # 如果来源是 data.xlsx，我们也可以存入，但前端不显示图片即可
            source_url = f"/images/{source_name}" if source_name.endswith(('.jpg', '.png')) else ""

            chunks = splitter.split_text(content)
            print(f"📖 正在处理文件 [{source_name}] 的 {len(chunks)} 个片段...")

            collection.upsert(
                ids=[f"text_{source_name}_{j}" for j in range(len(chunks))],
                documents=chunks,
                # 注意：这里千万不要传 uris 参数，否则会再次报 PIL 错误！
                metadatas=[{
                    "type": "text",
                    "source": source_name,
                    "img_url": source_url  # 将图片路径藏在元数据里
                } for _ in range(len(chunks))]
            )

def multimodal_search(query_text):
    print(f"\n搜索需求: '{query_text}'")
    results = collection.query(
        query_texts=[query_text],
        n_results=2,  # 返回前 2 名
        include=['documents', 'metadatas', 'uris', 'distances']
    )

    for i in range(len(results['ids'][0])):
        dist = results['distances'][0][i]
        meta = results['metadatas'][0][i]

        if meta['type'] == 'image':
            print(f"[图片匹配] 相似度: {1 - dist:.4f} | 路径: {results['uris'][0][i]}")
        else:
            doc = results['documents'][0][i]
            # 截取匹配到的文字片段
            print(f"[文本匹配] 相似度: {1 - dist:.4f} | 内容: {doc[:150]}...")



if __name__ == "__main__":
    # 第一次运行：构建大脑
    ingest_multimodal_data(".")

    # 测试搜索
    multimodal_search("那个绿色的护发素里含燕麦吗")
    multimodal_search("高性能服务器的价格")