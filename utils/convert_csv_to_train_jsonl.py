import csv
import json
import argparse
from tqdm import tqdm
import os


def convert_csv_to_jsonl(csv_path, output_path):
    data = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="转换中"):
            # 构造选项文本
            option_text = "\n".join([f"{opt}. {row[opt]}" for opt in ["A", "B", "C", "D"] if row[opt].strip()])

            # 构造用户问题
            user_content = f"{row['question']}\n{option_text}"

            # 构造 AI 回答
            assistant_content = f"答案是：{row['answer']}"
            if row.get("explanation") and row["explanation"].strip():
                assistant_content += f"。\n解析：{row['explanation'].strip()}"

            # 构造消息列表
            messages = [
                {"role": "system", "content": "你是一个专业的林业智能问答助手"},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]

            # 构造整体样本
            entry = {
                "id": row.get("id", ""),
                "eval_id": row.get("eval_id", ""),
                "knowledge": row.get("knowledge_point", ""),
                "question_type": "multiple_choice",
                "messages": messages
            }

            data.append(entry)

    # 保存为 JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f_out:
        for item in data:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 成功保存训练数据：{output_path}（共 {len(data)} 条）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="输入 CSV 文件路径")
    parser.add_argument("--out", type=str, required=True, help="输出 JSONL 文件路径")
    args = parser.parse_args()

    convert_csv_to_jsonl(args.csv, args.out)
