import pandas as pd


def parse_excel(file_path):
    # sheet_name=None 表示读取所有工作表
    excel_data = pd.read_excel(file_path, sheet_name=None)

    formatted_text = ""
    for sheet_name, df in excel_data.items():
        formatted_text += f"\n### 表格名称: {sheet_name}\n"

        # 处理空值，防止 AI 产生幻觉
        df = df.fillna("")

        # 核心：转成 Markdown 格式
        markdown_table = df.to_markdown(index=False)
        formatted_text += markdown_table + "\n"

    return formatted_text

# 测试（如果你手头有一个 Excel 的话）
print(parse_excel("data.xlsx"))