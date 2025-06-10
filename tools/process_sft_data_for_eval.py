import os
import sys
import json
import csv
import argparse
import pandas as pd
from tqdm import tqdm

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

def process_jsonl(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def save_csv(path, rows, header):
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(path, index=False, encoding="utf-8")

def extract_low_mastery_eval_set(jsonl_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    mc_rows = []  # 单选题 CSV
    qa_jsonl = []  # 问答题 JSONL
    current_id = 0  # 排序用新 ID
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="筛选评测数据"):
            try:
                item = json.loads(line.strip())
            except:
                continue

            item_id = item.get("id", "")
            questions = item.get("question_setter", {}).get("questions", [])
            evaluations = item.get("grading_teacher", {}).get("evaluations", [])

            for idx, q in enumerate(questions):
                mastery_level = evaluations[idx].get("evaluation", {}).get("mastery_level", "") if idx < len(evaluations) else ""
                if mastery_level != "l":
                    continue  # 只保留低掌握度

                q_type = q.get("question_type", "")
                knowledge = q.get("knowledge", "")
                response = q.get("response", {})

                base_info = {
                    "id": item_id,
                    "id_index": idx,
                    "question_type": q_type,
                    "knowledge": knowledge
                }

                if q_type == "multiple_choice":
                    if isinstance(response, str):
                        parts = [p.strip() for p in response.strip().split(",") if p.strip()]
                        if len(parts) >= 6:
                            question = parts[0]
                            options = parts[1:5]
                            answer = parts[5]
                            row = {
                                "id": f"{current_id}",
                                "question": question,
                                "A": options[0],
                                "B": options[1],
                                "C": options[2],
                                "D": options[3],
                                "answer": answer,
                                "explanation": "",
                                "item_id": f"{item_id}",
                                "knowledge_point": knowledge
                            }
                            mc_rows.append(row)
                            current_id += 1

                elif q_type in ["short_answer", "open_discussion"]:
                    if isinstance(response, dict):
                        qa_jsonl.append({
                            "id": item_id,
                            "question_type": q_type,
                            "knowledge": knowledge,
                            "history": [],
                            "query": response.get("question", ""),
                            "response": response.get("answer", "")
                        })

    save_csv(os.path.join(output_dir, "eval_multiple_choice.csv"), mc_rows,
             ["id", "question", "A", "B", "C", "D", "answer", "explanation", "item_id", "knowledge_point"])
    save_jsonl(os.path.join(output_dir, "eval_general_qa.jsonl"), qa_jsonl)
    print(f"✅ 已提取评测集：单选题 {len(mc_rows)} 条，问答题 {len(qa_jsonl)} 条")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/mnt/sda/wyp/forestllm-main/output/step5/merged_step5_dedup.jsonl", help="输入 JSONL 文件路径")
    parser.add_argument("--output_dir", type=str, default="/mnt/sda/wyp/forestllm-main/forest_eval/book", help="输出目录")
    args = parser.parse_args()

    extract_low_mastery_eval_set(args.input, args.output_dir)
