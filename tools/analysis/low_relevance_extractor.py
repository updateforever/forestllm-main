import json
import os
import logging
from datetime import datetime

# 配置日志
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f"/home/wyp/project/ForestLLM/outputs/logs/filter_{timestamp}.log",
            mode="w",
            encoding="utf-8",
        ),
    ],
)


def load_data(file_path):
    """
    加载 JSON 文件中的数据 (list of entries)
    """
    data = []
    if not os.path.exists(file_path):
        logging.error(f"文件 {file_path} 不存在!")
        return data

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            logging.info(f"成功加载 {len(data)} 条数据!")
        except json.JSONDecodeError as e:
            logging.error(f"加载数据时发生错误: {e}")
    return data


def save_data(data, file_path):
    """
    将处理后的数据保存回新的 JSON 文件
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"处理后的数据已保存到 {file_path}")
    except Exception as e:
        logging.error(f"保存数据时发生错误: {e}")


def filter_questions_across_fields(entry, threshold=6):
    """
    针对单个 entry，根据 expert_agent["refined_questions"] 的 relevance_score 删除对应索引的:
      - question_setter["questions"]
      - expert_agent["refined_questions"]
      - virtual_teacher["processed_results"]
    等字段的第 i 条内容 (若这些字段存在且长度足够)。

    说明:
      - 若某字段不存在，或长度不足，则仅在存在的字段中进行删除。
      - steps 通常是一个字典，比如 {"1": "completed", "2": "completed"}，并无按题目索引的记录，不作处理。
    """

    # 1) 获取必要字段; 如果不存在，则跳过（保持原样）
    qs = entry.get("question_setter", {})
    ea = entry.get("expert_agent", {})
    vt = entry.get("virtual_teacher", {})

    questions = qs.get("questions", [])
    refined_questions = ea.get("refined_questions", [])
    vt_results = vt.get("processed_results", [])

    # 若 refined_questions 为空，则无从判断 relevance_score，直接返回
    if not refined_questions:
        return entry

    # 新的列表，用于存放过滤后的结果
    filtered_questions = []
    filtered_refined_qs = []
    filtered_vt_results = []

    # 计算可遍历的最小长度(避免索引越界)
    min_len = min(len(questions), len(refined_questions), len(vt_results))

    # 如果 virtual_teacher["processed_results"] 比较少，也要保证不要越界
    # 如果比 question_setter / expert_agent 短，那末尾的问题自然也无 vt 对应项
    # 可以分两步处理：先过滤前 min_len，然后把没对上的尾部追加进新的列表（视需求而定）
    # 这里示范：仅对齐同样长度部分进行对比，尾部无法对齐的保留/丢弃可自行选择

    for i in range(min_len):
        rq = refined_questions[i]
        score = rq.get("relevance_score", 0)
        if score >= threshold:
            # 该试题保留
            filtered_questions.append(questions[i])
            filtered_refined_qs.append(rq)
            filtered_vt_results.append(vt_results[i])
        else:
            logging.info(
                f"entry ID={entry.get('id')} 第{i}条试题被删除: relevance_score={score} < {threshold}"
            )

    # 如果 question_setter / expert_agent / vt 有多余长度
    # (例如 vt_results 比其他的多几条)，可根据需求决定是否保留
    # 这里演示“保留其余未对齐部分”
    if len(questions) > min_len:
        filtered_questions.extend(questions[min_len:])
    if len(refined_questions) > min_len:
        filtered_refined_qs.extend(refined_questions[min_len:])
    if len(vt_results) > min_len:
        filtered_vt_results.extend(vt_results[min_len:])

    # 2) 回写过滤后的结果
    qs["questions"] = filtered_questions
    ea["refined_questions"] = filtered_refined_qs
    vt["processed_results"] = filtered_vt_results

    entry["question_setter"] = qs
    entry["expert_agent"] = ea
    entry["virtual_teacher"] = vt

    return entry


def process_data(input_file, output_file, threshold=6):
    """
    主处理流程：
      1. 读取数据
      2. 对每条数据进行过滤
      3. 将过滤结果写入新文件
    """
    data = load_data(input_file)

    processed_data = []
    for entry in data:
        processed_entry = filter_questions_across_fields(entry, threshold=threshold)
        processed_data.append(processed_entry)

    save_data(processed_data, output_file)


def main():
    # 路径示例，可根据需要修改
    input_path = "/home/wyp/project/ForestLLM/outputs/article/qwen_article_output_01_deduplicated.json"
    output_path = "/home/wyp/project/ForestLLM/outputs/article/qwen_article_output_02_lowrelevance_filtered.json"

    # 相关度阈值: 小于6则删除
    relevance_threshold = 6

    process_data(input_path, output_path, relevance_threshold)


if __name__ == "__main__":
    main()
