import gc
import os

import pandas as pd
from unstructured.partition.auto import partition
from paddleocr import PaddleOCR
import paddle

paddle.set_device('cpu')
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 1. 初始化 OCR 引擎 (全局初始化一次，节省时间) ---
# use_angle_cls=True 自动修正文字方向
# ocr = PaddleOCR(use_textline_orientation=True, lang="ch")


import numpy as np

import cn_clip.clip as clip
from chromadb.utils.embedding_functions import EmbeddingFunction

# --- 新增：自定义 Chinese-CLIP 嵌入类 ---
import os

import pandas as pd
from unstructured.partition.auto import partition
from paddleocr import PaddleOCR
import paddle

paddle.set_device('cpu')
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'


# --- 1. 初始化 OCR 引擎 (全局初始化一次，节省时间) ---
# use_angle_cls=True 自动修正文字方向
# ocr = PaddleOCR(use_textline_orientation=True, lang="ch")

from chromadb.utils.embedding_functions import EmbeddingFunction

import cn_clip.clip as clip
import paddle
from PIL import Image
import numpy as np


class ChineseCLIPEF(EmbeddingFunction):
    def __init__(self):
        # 加载中文 CLIP 模型
        self.model, self.preprocess = clip.load_from_name("ViT-B-16", device="cpu")
        self.model.eval()

    def __call__(self, input):
        if not input:
            return []

        # 情况 A: 如果输入是文字 (用于检索或存入 documents)
        if isinstance(input[0], str):
            text = clip.tokenize(input)
            with paddle.no_grad():
                text_features = self.model.encode_text(text)
                # 归一化，保证余弦相似度计算准确
                text_features /= text_features.norm(dim=-1, keepdim=True)
                return text_features.tolist()

        # 情况 B: 如果输入是图片 (用于 collection.upsert 时的 images 参数)
        else:
            image_features_list = []
            for item in input:
                # 将 numpy 数组转回 PIL 图片进行预处理
                pil_img = Image.fromarray(item) if isinstance(item, np.ndarray) else item
                image = self.preprocess(pil_img).unsqueeze(0)

                with paddle.no_grad():
                    image_features = self.model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    image_features_list.append(image_features.tolist()[0])
            return image_features_list

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
# multimodal_ef = OpenCLIPEmbeddingFunction(model_name=model_name, checkpoint=checkpoint)


multimodal_ef = ChineseCLIPEF()
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
        if not os.path.isfile(os.path.join(folder_path, item)):
            continue
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



def get_ocr_text(img_path):
    try:
        result = ocr.predict(img_path)
        if not result or len(result) == 0:
            return ""

        res_obj = result[0]
        texts = res_obj.get('rec_texts', [])
        scores = res_obj.get('rec_scores', [])

        # 组装有效文字（过滤掉分数过低的，比如你结果里那个分数 0.0 的空行）
        final_lines = []
        for text, score in zip(texts, scores):
            if score > 0.5 and text.strip():
                final_lines.append(text)

        gc.collect()
        return "\n".join(final_lines)
    except Exception as e:
        print(f"⚠️ OCR 解析 {img_path} 失败: {e}")
        return ""  # 失败时返回空字符串，不要把错误信息写进数据库

# --- 初始化参数更新 (适配 3.2.0) ---
ocr = PaddleOCR(
    use_textline_orientation=True,
    lang="ch",
    # 使用最新的参数名，确保设置生效
    text_det_limit_side_len=2500,
    text_det_thresh=0.1,  # 极致灵敏度
    text_det_box_thresh=0.1
)




def process_to_knowledge_base(input_folder, output_file):
    """核心逻辑：多模态内容提取与汇总"""
    all_files = [f for f in os.listdir(input_folder)]

    with open(output_file, "w", encoding="utf-8") as kb_file:
        for file_name in all_files:
            file_path = os.path.join(input_folder, file_name)

            # 跳过生成的输出文件本身和脚本文件
            if file_name in [output_file, "main.py"] or file_name.endswith(".py"):
                continue

            kb_file.write(f"\n{'=' * 20}\n")
            kb_file.write(f"📥 文件名: {file_name}\n")
            kb_file.write(f"{'=' * 20}\n\n")

            print(f"🚀 正在处理: {file_name}")

            try:
                # 情况 A: Word 和 PPT
                if file_name.endswith(('.docx', '.pptx','.html', '.htm')):
                    elements = partition(filename=file_path)
                    content = "\n".join([str(el) for el in elements])
                    kb_file.write(content + "\n")

                # 情况 B: Excel 表格
                elif file_name.endswith(('.xlsx', '.xls')):
                    df_dict = pd.read_excel(file_path, sheet_name=None)
                    for sheet_name, df in df_dict.items():
                        kb_file.write(f"--- 表格: {sheet_name} ---\n")
                        kb_file.write(df.fillna("").to_markdown(index=False) + "\n\n")

                # 情况 C: 图片 OCR
                elif file_name.endswith(('.jpg', '.png', '.jpeg')):
                    text = get_ocr_text(file_path)
                    kb_file.write(text + "\n")


                elif file_name.endswith(('.txt', '.md')):
                    # 使用 utf-8 读取，如果报错可以尝试 'gbk'
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    kb_file.write(text + "\n")

                else:
                    if content.strip() and "⚠️" not in content:
                        kb_file.write(content + "\n")

            except Exception as e:
                # kb_file.write(f"❌ 解析出错: {str(e)}\n")
                print(f"Error processing {file_name}: {e}")



# --- 执行 ---
if __name__ == "__main__":
    # 路径纠错：确保不进入递归
    current_dir = os.getcwd()
    kb_file = "final_knowledge_base.txt"

    # 步骤 1: 只跑解析（此时会占用大量内存运行 PaddleOCR）
    # 如果已经有这个文件了，可以跳过这一步
    if not os.path.exists(kb_file):
        print("🚀 正在执行第一阶段：文档解析与 OCR...")
        process_to_knowledge_base(current_dir, kb_file)

    # 步骤 2: 物理重启脚本或清空 OCR 对象
    # 为了保险，你可以手动注销 ocr 对象释放内存
    import gc

    if 'ocr' in globals():
        del ocr
    gc.collect()

    # 步骤 3: 跑向量入库（此时主要占用内存的是 Chinese-CLIP）
    print("🚀 正在执行第二阶段：向量入库...")
    ingest_multimodal_data(current_dir)



