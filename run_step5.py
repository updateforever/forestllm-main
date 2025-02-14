import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from agents import GradingTeacher

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_json(file_path):
    """加载 JSON 数据"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"文件 {file_path} 未找到，返回空列表")
        return []
    except json.JSONDecodeError:
        logging.error(f"文件 {file_path} 解析失败，请检查格式")
        return []


def save_json_incremental(file_path, new_entries):
    """
    以增量方式保存 JSON 文件：
    1. 读取现有数据（如果文件存在）。
    2. 合并新数据，并去重。
    3. 保存回文件，确保不会丢失已有数据。
    """
    try:
        existing_data = load_json(file_path)
        existing_data.extend(new_entries)

        # 维护原顺序的同时去重（确保 id + knowledge 唯一性）
        seen = set()
        unique_entries = []
        for entry in existing_data:
            key = get_unique_key(entry)
            if key not in seen:
                seen.add(key)
                unique_entries.append(entry)

        # 重新写入去重后的数据
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(unique_entries, f, ensure_ascii=False, indent=2)

    except Exception as e:
        logging.error(f"保存文件 {file_path} 时出错: {e}")


def get_unique_key(entry):
    """
    构造唯一标识，用于判断数据是否已完成。
    使用 `id` + `question_setter['knowledge']` 字段。
    """
    return f"{entry['id']}::{entry.get('question_setter', {}).get('knowledge', '')}"


def evaluate_answers(entry, grader):
    """处理单条数据的评估任务"""
    try:
        # 获取优化后的问题或原始问题
        expert_agent = entry.get("expert_agent", {})
        refined_question = expert_agent.get("refined_response", "").strip()
        original_question = entry["question_setter"].get("response", "").strip()

        # 确保问题有效，否则跳过处理
        question = refined_question if refined_question else original_question
        if not question:
            logging.warning(f"跳过 ID {entry['id']}，因为问题为空")
            return None

        # 获取学习者的答案
        learner_answer = entry["simulated_learner"].get("learner_answers", "")
        if not learner_answer:
            logging.warning(f"跳过 ID {entry['id']}，因为学习者答案为空")
            return None

        # 调用 GradingTeacher 评估 API
        evaluation = grader.evaluate_answer(None, question, learner_answer)

        # 更新数据
        entry["grading_teacher"] = {"evaluation": evaluation}
        entry["steps"]["5"] = "completed"

        logging.info(f"Step 5 completed for ID: {entry['id']}")
        return entry

    except Exception as e:
        logging.error(f"Error processing ID {entry['id']}: {e}")
        return None


def process_entries_multithreaded(input_path, output_path, grader, num_threads=5, batch_size=100):
    """多线程处理 Step 5 任务，每处理 batch_size 条数据保存一次"""

    # 加载已完成的数据
    completed_entries = load_json(output_path)
    completed_keys = {get_unique_key(entry) for entry in completed_entries}

    # 加载输入数据，并移除已完成的数据
    all_entries = load_json(input_path)
    pending_entries = [entry for entry in all_entries if get_unique_key(entry) not in completed_keys]

    logging.info(f"已完成 Step 5 条数: {len(completed_entries)}")
    logging.info(f"待处理 Step 5 条数: {len(pending_entries)}")

    processed_count = 0
    total_to_process = len(pending_entries)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_entry = {}

        for entry in pending_entries:
            future = executor.submit(evaluate_answers, entry, grader)
            future_to_entry[future] = entry

            # 每 batch_size 个任务进行一次保存
            if len(future_to_entry) >= batch_size:
                batch_results = []
                for future in as_completed(future_to_entry):
                    result = future.result()
                    if result:
                        batch_results.append(result)
                        processed_count += 1

                future_to_entry.clear()

                # **增量保存**，不会覆盖原数据
                save_json_incremental(output_path, batch_results)
                logging.info(f"已处理 {processed_count}/{total_to_process} 条数据")

        # 处理剩余的任务
        batch_results = []
        for future in as_completed(future_to_entry):
            result = future.result()
            if result:
                batch_results.append(result)
                processed_count += 1

        # **增量保存剩余数据**
        save_json_incremental(output_path, batch_results)

    logging.info("Step 5 评估任务已全部完成")


if __name__ == "__main__":
    input_file = "/home/wyp/project/ForestLLM/outputs/article/qwen_article_output_04_step4done.json"
    output_file = "/home/wyp/project/ForestLLM/outputs/article/qwen_article_output_05_step5done.json"

    # 初始化 GradingTeacher API 代理
    grader = GradingTeacher(model="qwen")

    # 开始多线程处理 Step 5
    process_entries_multithreaded(input_file, output_file, grader, num_threads=16, batch_size=16)
