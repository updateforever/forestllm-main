import json

def evaluate_from_jsonl(result_path):
    correct = 0
    total = 0

    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("correct") is True:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"✅ 样本总数: {total}")
    print(f"✅ 正确数: {correct}")
    print(f"✅ 准确率 (Accuracy): {accuracy:.4f}")

if __name__ == "__main__":
    result_path = "/mnt/sda/wyp/forestllm-main/outputs/eval/Qwen3-8B/forest_zero_shot_v1/results.jsonl"
    evaluate_from_jsonl(result_path)


