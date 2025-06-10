import json
import csv
import argparse
import os

def load_results(result_path):
    """加载评估结果 JSON 文件"""
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["prompts"], data["predictions"], data["references"]

def extract_sub_eval(result1_path, result2_path, output_csv):
    # 加载两个模型的结果
    prompts1, preds1, refs1 = load_results(result1_path)
    prompts2, preds2, refs2 = load_results(result2_path)

    assert len(preds1) == len(preds2) == len(refs1) == len(refs2), "结果数量不一致"

    sub_rows = []
    for i, (p1, p2, ref, prompt) in enumerate(zip(preds1, preds2, refs1, prompts1)):
        if p1 == ref and p2 != ref:
            user_msg = next((m["content"] for m in prompt if m["role"] == "user"), "")
            sub_rows.append({
                "index": i,
                "question": user_msg.replace("\n", " "),  # 简化格式
                "answer": ref
            })

    # 写入 CSV
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "question", "answer"])
        writer.writeheader()
        writer.writerows(sub_rows)

    print(f"✅ 已保存 {len(sub_rows)} 条子评估样本到: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result1", type=str, default='/mnt/sda/wyp/forestllm-main/outputs/eval_api/DS-Qwen-7B/eval_multiple_choice_filtered/20250425_013936/results.json', help="模型1评估结果 JSON 路径")
    parser.add_argument("--result2", type=str, default='/mnt/sda/wyp/forestllm-main/outputs/eval_api/MiniCPM3-4B/eval_multiple_choice_filtered/20250425_105747/results.json', help="模型2评估结果 JSON 路径")
    parser.add_argument("--output_csv", type=str, default="sub_data.csv", help="筛选后保存路径")
    args = parser.parse_args()

    extract_sub_eval(args.result1, args.result2, args.output_csv)
