import json
import csv
import argparse
import os

def load_results(jsonl_path):
    """加载 results.jsonl 文件，返回 id 到 record 的映射"""
    records = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            records[record["id"]] = record
    return records

def load_original_csv(csv_path):
    """加载原始数据集，返回 id 到行数据的映射"""
    id_to_row = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_str = row.get("id")
            if id_str is not None:
                id_to_row[int(id_str)] = row
    return id_to_row

def extract_subsets(model1_results, model2_results, model3_results):
    """提取四类子集的 id 列表"""
    all_correct_ids = []
    model1_correct_23_wrong_ids = []
    model1_wrong_23_correct_ids = []
    all_wrong_ids = []

    for id_, rec1 in model1_results.items():
        rec2 = model2_results.get(id_)
        rec3 = model3_results.get(id_)
        if not rec2 or not rec3:
            continue

        correct1 = rec1.get("correct", False)
        correct2 = rec2.get("correct", False)
        correct3 = rec3.get("correct", False)

        if correct1 and correct2 and correct3:
            all_correct_ids.append(id_)
        elif correct1 and not correct2 and not correct3:
            model1_correct_23_wrong_ids.append(id_)
        elif not correct1 and correct2 and correct3:
            model1_wrong_23_correct_ids.append(id_)
        elif not correct1 and not correct2 and not correct3:
            all_wrong_ids.append(id_)

    return all_correct_ids, model1_correct_23_wrong_ids, model1_wrong_23_correct_ids, all_wrong_ids


def save_subset_csv(selected_ids, original_rows, output_csv_path, fieldnames):
    """根据 id 列表筛选原始 CSV，并保存"""
    subset_rows = [original_rows[i] for i in selected_ids if i in original_rows]
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(subset_rows)

def extract_all_combinations(model1_results, model2_results, model3_results):
    """将所有样本按8种组合分类，返回：组合键 -> id列表"""
    combo_to_ids = {
        "111": [], "110": [], "101": [], "100": [],
        "011": [], "010": [], "001": [], "000": []
    }

    for id_, rec1 in model1_results.items():
        rec2 = model2_results.get(id_)
        rec3 = model3_results.get(id_)
        if not rec2 or not rec3:
            continue

        b1 = int(bool(rec1.get("correct", False)))
        b2 = int(bool(rec2.get("correct", False)))
        b3 = int(bool(rec3.get("correct", False)))

        key = f"{b1}{b2}{b3}"
        combo_to_ids[key].append(id_)

    return combo_to_ids

def merge_selected_combinations(combo_to_ids, original_rows, output_dir, fieldnames, selected_keys, output_filename):
    """将指定组合合并成一个新的测试集"""
    merged_rows = []

    for key in selected_keys:
        ids = combo_to_ids.get(key, [])
        for id_ in ids:
            if id_ in original_rows:
                row = dict(original_rows[id_])  # 拷贝一份
                merged_rows.append(row)

    merged_fieldnames = fieldnames

    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"✅ 合并后的新测试集 {output_filename}：{len(merged_rows)} 条样本")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1_result", type=str, required=True, help="模型1的 results.jsonl")
    parser.add_argument("--model2_result", type=str, required=True, help="模型2的 results.jsonl")
    parser.add_argument("--model3_result", type=str, required=True, help="模型3的 results.jsonl")
    parser.add_argument("--original_csv", type=str, required=True, help="原始完整评测数据集 CSV")
    parser.add_argument("--output_dir", type=str, default="outputs/compare_subsets", help="输出保存目录")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载
    model1_results = load_results(args.model1_result)
    model2_results = load_results(args.model2_result)
    model3_results = load_results(args.model3_result)
    original_rows = load_original_csv(args.original_csv)

    # 获取列头
    fieldnames = list(next(iter(original_rows.values())).keys())

    # 全组合提取
    combo_to_ids = extract_all_combinations(model1_results, model2_results, model3_results)

    # 按组合保存
    for key, id_list in combo_to_ids.items():
        filename = f"correct_{key}.csv"
        out_path = os.path.join(args.output_dir, filename)
        save_subset_csv(id_list, original_rows, out_path, fieldnames)
        print(f"✅ 已保存 {filename}：{len(id_list)} 条")

    print(f"📂 全部组合已保存到目录：{args.output_dir}")

    # 保存合并后的新测试集
    selected_keys = ["000", "100", "110", "101", "111"]
    merge_selected_combinations(
        combo_to_ids,
        original_rows,
        args.output_dir,
        fieldnames,
        selected_keys,
        output_filename="merged_testset.csv"
    )


if __name__ == "__main__":
    main()


"""
python extract_sub_data.py \
  --model2_result /mnt/sda/wyp/forestllm-main/outputs/eval/DeepSeek-R1-Distill-Qwen-7B/eval_multiple_choice_filtered/results.jsonl\
  --model1_result /mnt/sda/wyp/forestllm-main/outputs/eval/checkpoint-12277/eval_multiple_choice_filtered/results.jsonl \
  --model3_result /mnt/sda/wyp/forestllm-main/outputs/eval_api/gpt4/eval_multiple_choice_filtered/results.jsonl \
  --original_csv /mnt/sda/wyp/forestllm-main/forest_eval/book/eval_multiple_choice_filtered.csv \
  --output_dir /mnt/sda/wyp/forestllm-main/outputs/compare_subsets
"""