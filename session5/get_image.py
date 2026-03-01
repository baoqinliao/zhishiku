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


import numpy as np


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

        return "\n".join(final_lines)
    except Exception as e:
        return f"解析异常: {e}"

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
                    kb_file.write("⚠️ 不支持的文件格式，跳过处理。\n")

            except Exception as e:
                kb_file.write(f"❌ 解析出错: {str(e)}\n")
                print(f"Error processing {file_name}: {e}")

    print(f"\n✅ 任务完成！所有内容已汇总至: {output_file}")


# --- 执行 ---
if __name__ == "__main__":
    # 处理当前目录
    process_to_knowledge_base(".", "final_knowledge_base.txt")