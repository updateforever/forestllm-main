import os
import sys

# 添加项目根路径
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import json
import argparse
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from agents import GradingTeacher

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_jsonl(file_path: str) -> List[dict]:
    """加载 JSONL 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def save_jsonl_append(file_path: str, data: List[dict]):
    """将数据追加写入 JSONL 文件"""
    with open(file_path, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def get_existing_ids(file_path: str) -> set:
    """获取已存在的 ID 集合，避免重复处理"""
    if not os.path.exists(file_path):
        return set()
    with open(file_path, "r", encoding="utf-8") as f:
        return {json.loads(line).get("id") for line in f if line.strip()}

def evaluate_batch(batch: List[dict], grader: GradingTeacher) -> List[dict]:
    """评估一个批次的 entry"""
    for entry in batch:
        questions = entry.get("question_setter", {}).get("questions", [])
        refined_list = entry.get("expert_agent", {}).get("refined_questions", [])
        answers = entry.get("simulated_learner", {}).get("learner_answers", [])

        evaluations = []

        for i, q in enumerate(questions):
            # 选择优化题或原始题
            if (
                i < len(refined_list)
                and refined_list[i].get("requires_refinement", False)
                and refined_list[i].get("refined_response")
            ):
                question_text = refined_list[i]["refined_response"]
            else:
                question_text = q.get("response", "")
                if isinstance(question_text, dict):
                    question_text = '问题： ' + question_text.get("question", "") + '参考答案： ' + question_text.get("answer", "") 

            answer_text = answers[i].get("answer", "") if i < len(answers) else ""
            if not question_text or not answer_text:
                logging.warning(f"跳过空问题或答案: {entry['id']} - {i}")
                evaluations.append({"evaluation": None})
                continue
                
            result = grader.evaluate_answer(None, question_text, answer_text)
            evaluations.append({"evaluation": result})

        entry["grading_teacher"] = {"evaluations": evaluations}
        entry.setdefault("steps", {})["5"] = "completed"

    return batch

def run_step5(input_path: str, output_path: str, batch_size: int = 20):
    logging.info("🔍 加载数据...")
    all_data = load_jsonl(input_path)
    existing_ids = get_existing_ids(output_path)
    to_process = [e for e in all_data if e.get("id") not in existing_ids or e.get("steps", {}).get("5") != "completed"]

    logging.info(f"共加载 {len(all_data)} 条数据，待处理 {len(to_process)} 条")

    grader = GradingTeacher(model="qwen")

    for i in range(0, len(to_process), batch_size):
        batch = to_process[i : i + batch_size]
        processed = evaluate_batch(batch, grader)
        save_jsonl_append(output_path, processed)
        logging.info(f"✅ 已评估并保存 {i + len(batch)} 条")

    logging.info("🎉 Step 5 全部完成")

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default='/seu_share/home/luxiaobo/230248984/code/forestllm-main/output/merged_step4_dedup.jsonl', help="输入 JSONL 文件路径")
    parser.add_argument("--output_path", type=str, default='/seu_share/home/luxiaobo/230248984/code/forestllm-main/output/qwen_allbook_output_step5done.jsonl', help="输出 JSONL 文件路径")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    run_step5(args.input_path, args.output_path, args.batch_size)


# python step5_run.py \
#     --input_path /mnt/sda/wyp/forestllm-main/output/step5/part1.jsonl \
#     --output_path /mnt/sda/wyp/forestllm-main/output/step5/part1_step5.jsonl \
#     --batch_size 1
