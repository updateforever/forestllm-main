import json
import os
import sys
import logging
import argparse
# 添加项目根路径
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
from agents import SimulatedLearner
from typing import List

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def save_jsonl_append(file_path: str, data: List[dict]):
    with open(file_path, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def get_existing_ids(file_path: str) -> set:
    if not os.path.exists(file_path):
        return set()
    with open(file_path, "r", encoding="utf-8") as f:
        return {json.loads(line).get("id") for line in f if line.strip()}

def load_jsonl(file_path):
    """加载 JSONL 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def save_jsonl(file_path, data):
    """保存为 JSONL 文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logging.info(f"数据已保存至 {file_path}")

def process_step_4_batch(entries, learner):
    """批量处理 Step 4"""
    response_data_list = [entry["question_setter"] for entry in entries]
    batch_answers = learner.answer_questions_batch(response_data_list)

    for entry, answer in zip(entries, batch_answers):
        entry["simulated_learner"] = {"learner_answers": answer}
        steps = entry.get("steps", {})
        steps["4"] = "completed"
        entry["steps"] = steps

    logging.info(f"处理完成 {len(entries)} 条 Step 4 数据")
    return entries

def load_json_file_if_exists(file_path):
    """如果文件存在则加载，否则返回空列表"""
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"⚠️ 文件解析失败: {file_path}")
            return []

# ===== Step 4 处理逻辑 =====
def process_batch(entries: List[dict], learner: SimulatedLearner) -> List[dict]:
    processed_entries = []

    def extract_question(q_data):
        """从问答题结构中提取问题文本"""
        if isinstance(q_data, dict) and "question" in q_data:
            return q_data["question"]
        return str(q_data[:-2])  # fallback


    for entry in entries:
        question_list = entry.get("question_setter", {}).get("questions", [])
        refined_list = entry.get("expert_agent", {}).get("refined_questions", [])
        vt_list = entry.get("virtual_teacher", {}).get("processed_results", [])

        learner_inputs = []
        for i, q in enumerate(question_list):
            # 优先使用优化后的问题，否则使用原始问题
            if (
                i < len(refined_list)
                and refined_list[i].get("requires_refinement", False)
                and refined_list[i].get("refined_response")
            ):
                q_text = refined_list[i]["refined_response"]
            else:
                q_text = q.get("response", "")
                
            # 问答题需要解析出来问题
            q_text = extract_question(q_text)

            learner_inputs.append(q_text)

        # 批量生成答案
        batch_answers = learner.answer_questions_batch(learner_inputs)
        entry["simulated_learner"] = {"learner_answers": [{"answer": a} for a in batch_answers]}
        entry.setdefault("steps", {})["4"] = "completed"

        processed_entries.append(entry)

    return processed_entries

# ===== 主处理流程 =====
def run_step4(input_path: str, output_path: str, batch_size: int = 20):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    all_data = load_jsonl(input_path)
    existing_ids = get_existing_ids(output_path)
    to_process = [entry for entry in all_data if entry.get("id") not in existing_ids]

    logging.info(f"总数据：{len(all_data)}，未处理：{len(to_process)}，已完成：{len(existing_ids)}")

    learner = SimulatedLearner(
        model_api=["qwen"],
        model_paths=[
            "/home/wyp/project/swift/models/qwen25_7b_ins",
            "/home/wyp/project/swift/models/minicpm3-4b",
            "/home/wyp/project/swift/models/llama_3_1_8b_ins",
        ],
        model_platforms=["modelscope", "modelscope", "modelscope"],
    )

    for i in range(0, len(to_process), batch_size):
        batch = to_process[i : i + batch_size]
        processed = process_batch(batch, learner)
        save_jsonl_append(output_path, processed)
        logging.info(f"写入完成：{i + len(batch)} / {len(to_process)}")

    logging.info("✅ 所有 Step 4 推理完成")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/home/wyp/project/forest/forestllm-main/outputs/0321/qwen_book_output.jsonl", help="原始 jsonl 文件")
    parser.add_argument("--output_path", type=str, default="/home/wyp/project/forest/forestllm-main/outputs/0321/qwen_book_output_step4.jsonl", help="输出 jsonl 路径")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    run_step4(args.input_path, args.output_path, args.batch_size)

