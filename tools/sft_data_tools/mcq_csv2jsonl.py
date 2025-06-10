import csv
import json
import argparse

def convert_csv_to_jsonl(input_csv, output_jsonl):
    with open(input_csv, mode='r', encoding='utf-8') as csv_file, open(output_jsonl, mode='w', encoding='utf-8') as jsonl_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            question = row["question"].strip()
            options = {
                "A": row["A"].strip(),
                "B": row["B"].strip(),
                "C": row["C"].strip(),
                "D": row["D"].strip()
            }
            answer = row["answer"].strip().upper()
            explain = row.get("explain", "").strip()  # 可能没有explain

            # 生成user内容
            user_content = f"请阅读下列单选题，并在答案栏中只填写选择的字母，例如：\"答案\": \"C\"。\n单选题：{question}\nA) {options['A']}\nB) {options['B']}\nC) {options['C']}\nD) {options['D']} /no_think"

            # 生成assistant内容
            assistant_content = f"<think>\n\n</think>\n\n\"答案\": \"{answer}\""

            # 如果有解释，追加解释
            # if explain:
            #     assistant_content += f"\n解释：{explain}"

            record = {
                "messages": [
                    {"role": "system", "content": "你是一个专业的林业智能问答助手"},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
            }

            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ 转换完成，已保存为 JSONL 文件: {output_jsonl}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="输入的 CSV 文件路径")
    parser.add_argument("--output_jsonl", type=str, required=True, help="输出的 JSONL 文件路径")
    args = parser.parse_args()

    convert_csv_to_jsonl(args.input_csv, args.output_jsonl)

if __name__ == "__main__":
    main()

# python tools/sft_data_tools/mcq_csv2jsonl.py --input_csv /mnt/sda/wyp/forestllm-main/forest_eval/forest_zero_shot.csv --output_jsonl /mnt/sda/wyp/forestllm-main/forest_eval/forest_zero_shot.jsonl