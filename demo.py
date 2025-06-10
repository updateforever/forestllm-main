# from modelscope import AutoModelForCausalLM, AutoTokenizer

# model_name = "/mnt/sda/wyp/models/Qwen3-8B-Base"

# # load the tokenizer and the model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )

# # prepare the model input
# prompt = "Give me a short introduction to large language model."
# messages = [
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
#     enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# # conduct text completion
# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=32768
# )
# output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# # parsing thinking content
# try:
#     # rindex finding 151668 (</think>)
#     index = len(output_ids) - output_ids[::-1].index(151668)
# except ValueError:
#     index = 0

# thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
# content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

# print("thinking content:", thinking_content)
# print("content:", content)

import os
import csv
import argparse
from glob import glob
import re

# 主字段顺序（其余附加字段在后面原样保留）
MAIN_FIELDS = ["id", "question", "A", "B", "C", "D", "answer", "explanation", "source_file"]

def clean_option_text(text):
    """移除 A. B. A) B) 形式的前缀"""
    return re.sub(r"^[A-D][\.\)]\s*", "", text.strip())

def merge_csvs_with_cleaning(input_dir, output_csv, selected_keys=None):
    all_rows = []
    all_fields = set()  # 统计所有可能出现的字段
    csv_files = glob(os.path.join(input_dir, "*.csv"))

    print(f"🔍 找到 {len(csv_files)} 个 CSV 文件：")
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f" - {file_name}")
        file_key = file_name.replace("correct_", "").replace(".csv", "")

        if selected_keys and file_key not in selected_keys:
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cleaned_row = dict(row)  # 保留原始所有字段

                # 清洗选项
                cleaned_row["A"] = clean_option_text(row.get("A", ""))
                cleaned_row["B"] = clean_option_text(row.get("B", ""))
                cleaned_row["C"] = clean_option_text(row.get("C", ""))
                cleaned_row["D"] = clean_option_text(row.get("D", ""))

                # 标准化字段
                cleaned_row["question"] = row.get("question", "").strip()
                cleaned_row["answer"] = row.get("answer", "").strip().upper()
                cleaned_row["explanation"] = row.get("explanation", "").strip()
                cleaned_row["source_file"] = file_name

                all_rows.append(cleaned_row)
                all_fields.update(cleaned_row.keys())

    # 生成字段顺序（主字段 + 其余附加字段）
    additional_fields = sorted(set(all_fields) - set(MAIN_FIELDS))
    final_fields = MAIN_FIELDS + additional_fields

    # 重新编号 ID
    for i, row in enumerate(all_rows, 1):
        row["id"] = i

    # 写入 CSV
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"✅ 合并并清洗完成，共 {len(all_rows)} 条，字段数：{len(final_fields)}，已保存到：{output_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="包含多个CSV文件的目录")
    parser.add_argument("--output_csv", type=str, required=True, help="输出合并后的CSV文件路径")
    parser.add_argument("--selected_keys", nargs="*", help="仅合并指定key命名的CSV（如000 111）")
    args = parser.parse_args()

    merge_csvs_with_cleaning(args.input_dir, args.output_csv, selected_keys=args.selected_keys)

if __name__ == "__main__":
    main()


"""
python demo.py \
  --input_dir /mnt/sda/wyp/forestllm-main/outputs/compare_subsets/ \
  --output_csv /mnt/sda/wyp/forestllm-main/outputs/compare_subsets/merged_cleaned.csv \
  --selected_keys 000 100 111 101 110
"""