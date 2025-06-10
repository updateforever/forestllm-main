import os
import csv
import argparse
from glob import glob

# 固定字段顺序（必须按这个顺序输出）
TARGET_FIELDS = ["id", "question", "A", "B", "C", "D", "answer", "explain", "source_file"]

def merge_csvs_fixed(input_dir, output_csv):
    all_rows = []
    csv_files = glob(os.path.join(input_dir, "*.csv"))
    print(f"🔍 找到 {len(csv_files)} 个 CSV 文件：")
    for file in csv_files:
        print(f" - {file}")

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                merged_row = {
                    "question": row.get("question", "").strip(),
                    "A": row.get("A", "").strip(),
                    "B": row.get("B", "").strip(),
                    "C": row.get("C", "").strip(),
                    "D": row.get("D", "").strip(),
                    "answer": row.get("answer", "").strip().upper(),  # answer标准化为大写
                    "explain": row.get("explain", "").strip(),       # 有就用，没有就空
                    "source_file": file_name
                }
                all_rows.append(merged_row)

    # 🔥 重新编号 ID，从1开始
    for idx, row in enumerate(all_rows, start=1):
        row["id"] = idx

    # 💾 保存到目标文件
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TARGET_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"✅ 成功合并 {len(all_rows)} 条记录，保存到: {output_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="输入 CSV 文件夹路径")
    parser.add_argument("--output_csv", type=str, required=True, help="输出合并后的 CSV 文件路径")
    args = parser.parse_args()

    merge_csvs_fixed(args.input_dir, args.output_csv)

if __name__ == "__main__":
    main()



"""
python merge_csvs_lazy.py \
    --input_dir /mnt/sda/wyp/forestllm-main/forest_eval \
    --output_csv /mnt/sda/wyp/forestllm-main/forest_eval/merged_eval_lazy.csv

"""