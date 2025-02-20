import json
import logging
import argparse
from agents import (
    QuestionSetter,
    ExpertAgent,
    VirtualTeacherAgent,
    SimulatedLearner,
    GradingTeacher,
)

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_json(file_path):
    """加载JSON数据"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(file_path, data):
    """保存JSON数据"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logging.info(f"数据已保存至 {file_path}")

def process_step_4_batch(entries, learner):
    """批量处理 step 4 (Simulated Learner)"""
    response_data_list = [entry["question_setter"] for entry in entries]
    batch_answers = learner.answer_questions_batch(response_data_list)

    for entry, answer in zip(entries, batch_answers):
        entry["simulated_learner"] = {"learner_answers": answer}
        entry["steps"]["4"] = "completed"  # 这里改为字符串，避免 set 类型问题

    logging.info(f"批处理完成 {len(entries)} 条 Step 4 数据")
    return entries

def process_entries(input_path, output_path, learner, batch_size=40):
    """主流程，逐批处理未完成的 Step 4，并在每个批次后保存"""
    data = load_json(input_path)
    
    # 筛选出未完成 step 4 的数据
    pending_entries = [entry for entry in data if entry["steps"].get("4") != "completed"]
    completed_entries = [entry for entry in data if entry["steps"].get("4") == "completed"]

    logging.info(f"待处理 Step 4 条数: {len(pending_entries)}")

    # 批处理
    for i in range(0, len(pending_entries), batch_size):
        batch = pending_entries[i: i + batch_size]
        processed_batch = process_step_4_batch(batch, learner)
        completed_entries.extend(processed_batch)

        # **每个批次处理完就保存**
        save_json(output_path, completed_entries)
        logging.info(f"已保存进度：{len(completed_entries)} 条数据")

    logging.info("所有 Step 4 处理完成并保存")

def main(input_file, output_file):
    """程序入口"""
    # 初始化代理
    learner = SimulatedLearner(
        model_api=["qwen"],
        model_paths=[
            "/home/wyp/project/swift/models/qwen25_7b_ins",
            "/home/wyp/project/swift/models/minicpm3-4b",
            "/home/wyp/project/swift/models/llama_3_1_8b_ins",
        ],
        model_platforms=["modelscope", "modelscope", "modelscope"],
    )

    process_entries(input_file, output_file, learner, batch_size=40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理 Step 4 的数据")
    parser.add_argument("input_file", type=str, help="输入文件路径")
    parser.add_argument("output_file", type=str, help="输出文件路径")
    
    args = parser.parse_args()

    main(args.input_file, args.output_file)


    # input_file = "/home/wyp/project/ForestLLM/outputs/article/qwen_article_output_03_transformed.json"
    # output_file = "/home/wyp/project/ForestLLM/outputs/article/qwen_article_output_04_step4done.json"

