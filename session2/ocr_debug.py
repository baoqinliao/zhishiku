import os
from paddleocr import PaddleOCR
import paddle

# 1. 初始化引擎：强制开启“高灵敏度”模式
ocr = PaddleOCR(
    use_textline_orientation=True,
    lang="ch",
    # 调优参数：增加检测边长限制，防止大图小字被忽略
    det_limit_side_len=2500,
    # 调优参数：降低检测阈值，让 AI 更“敏感”
    det_db_thresh=0.2,
    det_db_box_thresh=0.2,
)


def debug_ocr(img_path):
    print(f"🔍 正在深度扫描: {img_path}")
    result = ocr.predict(img_path)

    if not result or len(result) == 0:
        print("❌ 引擎未返回结果")
        return

    # print(result)

    # 1. 获取第一个结果字典/对象
    res_obj = result[0]

    # 2. 直接提取文本列表和分数列表
    # 在 3.2.0 中，数据是平铺存储的，不再是嵌套在 line 里的
    texts = res_obj.get('rec_texts', [])
    scores = res_obj.get('rec_scores', [])
    boxes = res_obj.get('rec_boxes', [])

    print(f"✅ 成功找到 {len(texts)} 条文本记录：\n")

    # 3. 循环打印
    for i in range(len(texts)):
        content = texts[i]
        score = scores[i]
        # 获取坐标（可选）
        box = boxes[i] if i < len(boxes) else "无坐标"

        print(f"[{i:02d}] 内容: {content}")
        print(f"     信心: {score:.4f} | 坐标: {box}")
        print("-" * 20)


if __name__ == "__main__":
    debug_ocr("11.jpg")
