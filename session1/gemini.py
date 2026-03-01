import requests
import base64

# --- 配置区 ---
API_KEY = "AIzaSyBcIFwSMCskon8ies0ySfXyIbRammaQzxs"
IMAGE_PATH = "test.jpg"
# 接口地址（注意这是 Google 的标准地址）
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

base64_image = encode_image(IMAGE_PATH)

# Gemini 的数据结构与 OpenAI 稍有不同
payload = {
    "contents": [{
        "parts": [
            {"text": "请详细描述这张图片的内容。"},
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64_image
                }
            }
        ]
    }]
}

response = requests.post(URL, json=payload)

# 解析结果
try:
    description = response.json()['candidates'][0]['content']['parts'][0]['text']
    with open("description.txt", "w", encoding="utf-8") as f:
        f.write(description)
    print("成功！描述已保存。")
except Exception as e:
    print(f"解析失败: {response.json()}")