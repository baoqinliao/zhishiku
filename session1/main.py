import requests
import base64

# --- 配置区 ---
API_KEY = "sk-8710d0df0cd14841b49296965e605eb1"
IMAGE_PATH = "test.jpg"  # 确保你目录下有一张名为 test.jpg 的图
OUTPUT_FILE = "description.txt"

# 1. 编码图片
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

base64_image = encode_image(IMAGE_PATH)

# 2. 构造请求
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

payload = {
    "model": "deepseek-ai/deepseek-vl-7b-chat", # 或者 gpt-4o-mini
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请详细描述这张图片的内容。"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]
}

# 3. 发送请求并处理响应
response = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=payload)
print(response.text)
description = response.json()['choices'][0]['message']['content']

# 4. 保存为 txt
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(description)

print(f"成功！描述已保存至 {OUTPUT_FILE}")