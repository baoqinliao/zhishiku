import base64
import os
import chromadb
from llama_cpp import Llama
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from llama_cpp.llama_chat_format import Qwen25VLChatHandler# 确保安装了最新版 llama-cpp-python

from difflib import SequenceMatcher

import re



# --- 1. 启动真正的“视觉大脑” ---
MODEL_PATH = "../models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
MMPROJ_PATH = "../models/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"
TEXT_MODEL_PATH = "../models/qwen2.5-7b-instruct-q4_k_m.gguf"

# 初始化 Qwen2.5 特有的视觉句柄
chat_handler = Qwen25VLChatHandler(clip_model_path=MMPROJ_PATH)
# --- 1. 升级：启动多模态大脑 (Vision LLM) ---
# 注意：你需要安装适配器处理类


llm_vision = Llama(
    model_path=MODEL_PATH,
    chat_handler=chat_handler,
    n_ctx=2048,      # CPU 建议初始值，若内存充足可调至 4096
    n_threads=8,     # 充分利用 CPU 核心
    logits_all=True
)

# 1. 启动大脑 (LLM)
llm = Llama(
    model_path=TEXT_MODEL_PATH,
    n_ctx=4096,  # 调大一点，方便读取更多背景
    n_threads=4
)



# --- 辅助函数：将图片转为 Data URL ---
def image_to_base64_data_url(image_path):
    with open(image_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"



def calculate_info_overlap(text1, text2):
    """计算 AI 回答中的字符有多少比例在原文本中出现过"""
    if not text1 or not text2: return 0
    # 清理非字符内容，转换为集合
    set1 = set(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]', text1))
    set2 = set(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]', text2))
    if not set1: return 0
    # 计算交集在 AI 回答中的占比
    return len(set1.intersection(set2)) / len(set1)


def check_numbers_match(answer, source):
    """
    逻辑：AI 回答中提到的【所有数字】，必须能在【原文本】中找到。
    如果 AI 提到了原文本没有的数字，说明可能找错图了或 AI 在胡编。
    """
    ans_nums = set(re.findall(r'\d+', answer))
    src_nums = set(re.findall(r'\d+', source))

    if not ans_nums:
        return True  # AI 没提数字，默认通过

    # 检查 ans_nums 是否是 src_nums 的子集
    # 即：AI 说的数字，原文里全都有
    return ans_nums.issubset(src_nums)


def verify_answer_factuality(answer, context_text):
    """
    检查 AI 回答中的数字是否全部来自于上下文
    返回：(是否通过, 错误的数字列表)
    """
    # 提取 AI 回答中的所有数字（包括带小数点的）
    ans_nums = set(re.findall(r'\d+\.?\d*', answer))
    # 提取上下文中的所有数字
    ctx_nums = set(re.findall(r'\d+\.?\d*', context_text))

    # 找出那些 AI 说了，但原文里根本没有的数字
    hallucinated_nums = ans_nums - ctx_nums

    # 过滤掉一些常见的干扰项（比如 AI 可能会说“第1点”、“首先”等）
    # 如果数字很小且在 1-5 之间，可以考虑忽略，或者保留严格校验
    real_hallucinations = [n for n in hallucinated_nums if len(n) > 0]

    if not real_hallucinations:
        return True, []
    else:
        return False, real_hallucinations

# 1. 启动大脑 (LLM)

# 2. 启动记忆 (ChromaDB)
# 注意：这里需要和你第三阶段的模型参数保持完全一致
multimodal_ef = OpenCLIPEmbeddingFunction(
    model_name="ViT-B-32-quickgelu",
    checkpoint="laion400m_e32"
)
client = chromadb.PersistentClient(path="./my_multimodal_db")
collection = client.get_or_create_collection(
    name="multimodal_knowledge",
    embedding_function=multimodal_ef
)

import re


def get_hallucinated_numbers(answer, context_text):
    """
    找出 AI 回答中存在但在上下文里没出现的数字
    """
    # 提取数字（支持整数和小数）
    ans_nums = set(re.findall(r'\d+\.?\d*', answer))
    ctx_nums = set(re.findall(r'\d+\.?\d*', context_text))

    # 排除掉 AI 常见的列表编号（如 1. 2. 3.），这些通常不是事实数据
    common_indices = {'1', '2', '3', '4', '5'}
    hallucinated = (ans_nums - ctx_nums) - common_indices

    return list(hallucinated)

def integrated_ask(query,uploaded_image_b64=None):
    # --- 步骤 A: 检索 ---
    print(f"🔍 正在从知识库寻找线索...")
    n =20
    if uploaded_image_b64:
        n = 5


    results = collection.query(query_texts=[query], n_results=n, # 搜 5 个，增加图片入选的概率
        include=['documents', 'metadatas', 'uris', 'distances'])

    print(results)

    # 提取文本片段和图片路径


    context_list = []
    target_img_path = None
    source_urls = set()  # 使用集合去重
    SIMILARITY_THRESHOLD = 0.95
    IMAGE_DIST_THRESHOLD = 1.5

    # 临时存储检索到的文本和图片对应关系
    retrieved_candidates = []


    for i in range(len(results['ids'][0])):
        dist = results['distances'][0][i]
        meta = results['metadatas'][0][i]

        if dist > SIMILARITY_THRESHOLD:
            continue

        # 1. 如果搜到的是直接的图片条目
        if meta.get('type') == 'image' and dist < IMAGE_DIST_THRESHOLD:
            # 这里的 uri 是我们在 A 段存入的绝对路径，转换成 web 路径
            filename = os.path.basename(results['uris'][0][i])
            source_urls.add(f"/images/{filename}")

        # 2. 如果搜到的是文本条目
        elif meta.get('type') == 'text':
            context_list.append(results['documents'][0][i])

            # 看看这个文本片段是不是来自于某个图片 (如 11.jpg)
            img_url = meta.get('img_url')
            print(results['documents'][0][i],img_url,meta.get('source') )
            if img_url:
                # 只要有文本内容，就存入候选，用于后续比对
                retrieved_candidates.append({
                    "text": results['documents'][0][i],
                    "img_url": img_url
                })
                # source_urls.add(img_url)
    # --- 步骤 B: 生成 ---

    final_image_url = None
    mode = "text"
    context_text = "\n".join(context_list)

    if uploaded_image_b64:
        # 情况 1: 用户直接传了图 (Gemini 模式)
        final_image_url = uploaded_image_b64
        mode = "vision"

    if mode == "vision":
        print(f"🚀 视觉模式启动 | 来源: {'用户上传' if uploaded_image_b64 else '本地知识库'}")
        messages = [
            {"role": "system", "content": "你是一个全能助手。请结合图片和背景文字回答。"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": final_image_url}},
                    {"type": "text", "text": f"背景资料：\n{context_text}\n\n问题：{query}"}
                ]
            }
        ]
        response = llm_vision.create_chat_completion(messages=messages, max_tokens=512)
        answer = response["choices"][0]["message"]["content"]
    else:
        prompt = f"""<|im_start|>system
    你是一个专业的私有知识库助手。请根据已知信息，用简洁、专业的口吻回答用户问题。
    如果信息不足，请委婉告知。<|im_end|>
    <|im_start|>user
    已知信息：
    {context_text}
    
    问题：{query}<|im_end|>
    <|im_start|>assistant
    """

        print("🤖 AI 正在整合答案...")
        response = llm(prompt, max_tokens=512, stop=["<|im_end|>"])
        answer = response["choices"][0]["text"]

    # --- 步骤 C: 二次校验（图片过滤） ---
    final_source_urls = set()
    MATCH_THRESHOLD = 0.7  # 你要求的 0.8 相似度

    if not uploaded_image_b64:
        for cand in retrieved_candidates:
            if cand['img_url']:
                # 计算 AI 答案与原始检索文本的相似度
                sim_score = calculate_info_overlap(answer, cand['text'])
                print("888",sim_score,answer,cand['text'])

                # 只有当 AI 回答的内容与该片段高度相关时，才展示该片段关联的图片
                # 或者：如果 AI 答案中包含了检索片段的关键关键词
                if sim_score >= MATCH_THRESHOLD or cand['text'] in answer:
                    final_source_urls.add(cand['img_url'])

        wrong_numbers = get_hallucinated_numbers(answer, context_text)

        # 如果发现了“来历不明”的数字
        if wrong_numbers:
            # 拼接到答案末尾，用明显的样式标注
            warning_msg = f"\n\n⚠️ **数据声明**：检测到回答中包含原文未明确提取的数值 ({', '.join(wrong_numbers)})，请以随附的原始资料或图片为准。"
            answer += warning_msg




    return {
        "answer": answer,
        "sources": list(final_source_urls)
    }


if __name__ == "__main__":
    res = integrated_ask("那个绿色的护发素里含燕麦吗？多少钱？")
    print(f"\n【最终回答】:\n{res['answer']}")
    print(f"【参考图片】: {res['sources']}")