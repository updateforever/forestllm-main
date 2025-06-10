import os
import sys
import json
# 添加项目根路径
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
import pandas as pd
import argparse
from agents import VirtualTeacherAgent

# 初始化模型
virtual_teacher_r1 = VirtualTeacherAgent(model="deepseek-r1")
virtual_teacher = VirtualTeacherAgent(model="qwen")

def process_jsonl(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def process_book_data(jsonl_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    original_mc = []   # 原始选择题格式
    conversational_mc = []  # 口语化格式（无 CoT）
    conversational_mc_cot = []  # 口语化格式（有 CoT）
    general_qa_dialogue = []  # 普通问答对话格式
    general_qa_cot = []       # 带 CoT 思维链格式

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
            except:
                continue

            item_id = item.get("id", "")
            questions = item.get("question_setter", {}).get("questions", [])
            mastery_levels = item.get("grading_teacher", {}).get("evaluations", [])
            cot_list = item.get("virtual_teacher", {}).get("processed_results", [])
            answers = item.get("simulated_learner", {}).get("learner_answers", [])

            for idx, q in enumerate(questions):
                question_type = q.get("question_type", "")
                knowledge = q.get("knowledge", "")
                response = q.get("response", "")
                mastery_level = mastery_levels[idx].get("evaluation", {}).get("mastery_level", "unknown") if idx < len(mastery_levels) else "unknown"
                cot = cot_list[idx].get("CoT", "") if idx < len(cot_list) else ""
                conversational = cot_list[idx].get("conversational_form", {}) if idx < len(cot_list) else {}
                # learner_answer = answers[idx].get("answer", "") if idx < len(answers) else ""

                base = {
                    "id": item_id,
                    "id_index": idx,
                    "question_type": question_type,
                    "mastery_level": mastery_level,
                    "knowledge": knowledge
                }

                if question_type == "multiple_choice":
                    # 原始格式
                    original_mc.append({**base, "mcq": response})

                    # 口语化格式
                    if isinstance(conversational, dict) and conversational.get("question"):
                        conversational_mc.append({**base, "messages": [
                            {"role": "system", "content": "你是一个专业的林业智能问答助手"},
                            {"role": "user", "content": conversational["question"]},
                            {"role": "assistant", "content": conversational["answer"]},
                        ]})
                        # 口语化格式（带 CoT）
                        if cot:
                            conversational_mc_cot.append({**base, "messages": [
                                {"role": "system", "content": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"},
                                {"role": "user", "content": conversational["question"]},
                                {"role": "assistant", "content": f"<think>{cot}</think><answer>{conversational['answer']}</answer>"},
                            ]})

                elif question_type in ["short_answer", "open_discussion"]:
                    # COT格式
                    answer_text = response.get("answer") if isinstance(response, dict) else ""
                    question_text = response.get("question") if isinstance(response, dict) else ""
                    if cot:
                        full_answer = f"<think>{cot}</think><answer>{answer_text}</answer>"
                    else:
                        full_answer = answer_text

                    general_qa_cot.append({**base, "messages": [
                        {"role": "system", "content": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"},
                        {"role": "user", "content": question_text},
                        {"role": "assistant", "content": full_answer},
                    ]})

                    # 普通问答格式
                    general_qa_dialogue.append({**base, "messages": [
                        {"role": "system", "content": "你是一个专业的林业智能问答助手"},
                        {"role": "user", "content": question_text},
                        {"role": "assistant", "content": answer_text},
                    ]})


    # 输出保存
    save_jsonl(os.path.join(output_dir, "train_multiple_choice_original.jsonl"), original_mc)
    save_jsonl(os.path.join(output_dir, "train_multiple_choice_conversational.jsonl"), conversational_mc)
    save_jsonl(os.path.join(output_dir, "train_multiple_choice_conversational_cot.jsonl"), conversational_mc_cot)
    save_jsonl(os.path.join(output_dir, "train_general_qa_dialogue.jsonl"), general_qa_dialogue)
    save_jsonl(os.path.join(output_dir, "train_general_qa_cot.jsonl"), general_qa_cot)
    print(f"✅ 数据转换完成，共计选择题 {len(original_mc)}，简答题 {len(general_qa_dialogue)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='/mnt/sda/wyp/forestllm-main/output/step5/merged_step5_dedup.jsonl', help="输入 JSONL 文件路径")
    parser.add_argument("--output_dir", type=str, default="output/sft_data/book_split", help="输出目录")
    args = parser.parse_args()

    process_book_data(args.input, args.output_dir)
