import os
import sys
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import json
import hashlib
import logging
import argparse
from queue import Queue, Empty
from threading import Thread, Event
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from agents import (
    QuestionSetter,
    ExpertAgent,
    VirtualTeacherAgent,
    SimulatedLearner,
    GradingTeacher,
)
from utils.toolkit import clean_book_text, filter_web_text
from datetime import datetime
import time

# 获取当前时间，格式化为文件名友好的字符串
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f"/home/wyp/project/forest/forestllm-main/outputs/logs/process_{timestamp}.log",
            mode="w",
            encoding="utf-8",
        ),
    ],
)


# ===== 多线程操作 ===== #
# 保存数据的线程函数
def data_saver(queue: Queue, output_file: str, stop_event: Event, batch_size: int = 5):
    """
    从队列中读取数据并批量保存到 JSONL 文件。
    文件写入采用“读-改-写”策略，支持根据 ID 更新 steps。
    
    :param queue: 存储数据的线程安全队列
    :param output_file: 输出 JSONL 文件路径
    :param stop_event: 停止信号，用于安全关闭线程
    :param batch_size: 每次写入的最小数据量
    """
    buffer = []

    while not stop_event.is_set() or not queue.empty():
        try:
            data = queue.get(timeout=0.1)
            buffer.append(data)
            queue.task_done()
        except Empty:
            continue

        if len(buffer) >= batch_size:
            _write_to_file(output_file, buffer)
            buffer.clear()

    # 程序结束时写入剩余数据
    if buffer:
        _write_to_file(output_file, buffer)


def _write_to_file(output_file: str, new_data: list):
    """
    将数据写入 .jsonl 文件，直接追加写入，不做去重。
    后续可通过独立脚本去重。
    """
    with open(output_file, "a", encoding="utf-8") as f:
        for entry in new_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ==== tools ==== #
def load_data(data_file):
    """从JSONL文件中加载数据"""
    data = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    logging.info(f"加载了 {len(data)} 条数据")
    return data


def infer_data_class(data_file):
    """
    从数据文件的第一条数据中推断 data_class。
    :param data_file: JSONL 文件路径
    :return: 推断出的 data_class 或 'unknown'
    """
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                logging.warning("数据文件为空，无法推断 data_class")
                return "unknown"

            first_entry = json.loads(first_line)
            if "meta_info" not in first_entry:
                data_class = first_entry.get("class", "unknown")
                logging.info(f"自动推断到 data_class: {data_class}")
                return data_class
            else:
                data_class = first_entry.get("meta_info", {}).get(
                    "data_class", "unknown"
                )
                logging.info(f"自动推断到 data_class: {data_class}")
                return data_class
    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        logging.error(f"推断 data_class 时出错: {e}")
        return "unknown"

# 加载存在数据
def load_existing_data(out_folder, model_name, data_class):
    """
    兼容从 .json/.jsonl 文件读取已有数据，统一输出为 jsonl。
    """
    if out_folder.endswith("jsonl"): 
        out_file = out_folder  # 直接是完整的 jsonl 文件路径
    else:
        out_file = os.path.join(out_folder, f"{model_name}_{data_class}_output.jsonl")  # 自动拼接命名

    id_set = set()
    existing_data = []

    if os.path.exists(out_file):
        logging.info(f"加载已有处理数据文件: {out_file}")
        with open(out_file, "r", encoding="utf-8") as f:
            existing_data = [json.loads(line.strip()) for line in f if line.strip()]

    # 提取已存在的 id
    for entry in existing_data:
        if "id" in entry:
            id_set.add(entry["id"])

    logging.info(f"已有数据数量: {len(existing_data)}，将继续保存到: {out_file}")
    return out_file, id_set, existing_data



# 检查当前数据是否存在
def find_entry_by_id(out_file, entry_id):
    """
    在已保存的 JSON 文件中查找指定 ID 的数据条目，包含步骤状态信息。
    :param out_file: 保存数据的 JSON 文件路径
    :param entry_id: 要查找的唯一 ID
    :return: 找到的完整数据条目（包括 steps 信息），未找到则返回 {"id": entry_id, "steps": {}}
    """
    if not os.path.exists(out_file):
        logging.warning(f"文件不存在: {out_file}")
        return {"id": entry_id, "steps": {}}

    with open(out_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("id") == entry_id:
                return entry

    return {"id": entry_id, "steps": {}}


def preprocess_text(entry):
    """
    根据数据类型（data_class）对文本进行预处理
    """
    data_class = entry.get("class", "")
    text = entry.get("text", "")
    # 关键字列表，用于判断低质量数据
    skip_keywords = ["台湾", "毒", "广告", "稿"]

    if data_class == "article":
        MAX_TEXT_LENGTH = 10000
        text = text[:MAX_TEXT_LENGTH]
        for keyword in skip_keywords:
            if keyword in text:
                return None

    elif data_class == "web":
        text = filter_web_text(entry)
        if text is None:
            # logging.info("跳过低价值 Web 数据")
            return None

    elif data_class == "book":
        MAX_TEXT_LENGTH = 10000
        text = clean_book_text(text, max_length=MAX_TEXT_LENGTH)
        # logging.info(f"书籍类数据已清洗并截断到 {MAX_TEXT_LENGTH} 字符")

    else:
        logging.warning(f"未识别的数据类型: {data_class}，使用原始文本")

    return text


# 生成唯一 ID
def generate_entry_id(entry):
    entry_str = json.dumps(entry, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(entry_str.encode("utf-8")).hexdigest()


# 1️⃣ **Question Setter**
def process_question_setter(entry, question_setter):
    """
    直接处理 QuestionSetter 逻辑。
    :param entry: 单条数据条目
    :param question_setter: QuestionSetter 实例
    """
    entry_id = entry["id"]  # 使用 entry 中的 ID
    questions = question_setter.generate_response(
        entry["text"], entry.get("class", "")
    )  # 生成问题

    # 添加处理结果到 entry
    entry["question_setter"] = {"questions": questions}
    # logging.info(f"QuestionSetter 处理完成: {entry_id}")

    return entry


# 2️⃣ **Expert Agent**
def process_expert_agent(entry, expert_agent):
    """
    直接处理 ExpertAgent 逻辑。
    :param entry: 单条数据条目
    :param expert_agent: ExpertAgent 实例
    """
    entry_id = entry["id"]  # 使用 entry 中的 ID
    refined_questions = [
        expert_agent.evaluate_and_refine_question(
            entry["text"], q, entry.get("class", "")
        )
        for q in entry["question_setter"]["questions"]
    ]

    # 添加处理结果到 entry
    entry["expert_agent"] = {"refined_questions": refined_questions}
    # logging.info(f"ExpertAgent 处理完成: {entry_id}")

    return entry


# 3️⃣ **Virtual Teacher**
def process_virtual_teacher(entry, virtual_teacher):
    """
    直接处理 VirtualTeacher 逻辑，确保数据顺序一致，无需额外保存索引。
    :param entry: 单条数据条目
    :param virtual_teacher: VirtualTeacher 实例
    """
    entry_id = entry["id"]  # 使用 entry 中的 ID
    processed_results = []

    # 获取 question_setter 和 refined_questions
    questions = entry["question_setter"]["questions"]
    refined_questions = entry.get("expert_agent", {}).get("refined_questions", [])

    # 遍历试题，根据索引处理优化试题或原始试题
    for index, question_data in enumerate(questions):
        # 判断是否有优化后的试题
        if index < len(refined_questions) and refined_questions[index].get(
            "requires_refinement", False
        ):
            current_question = refined_questions[index]["refined_response"]
        else:
            current_question = question_data["response"]

        # 判断题型，执行相应处理
        if question_data.get("question_type") == "multiple_choice":
            conversational_form = virtual_teacher.convert_to_conversational_form(
                entry["text"], current_question, entry.get("class", "")
            )
            cot = virtual_teacher.generate_thinking_chain(
                entry["text"], conversational_form, entry.get("class", "")
            )
        else:
            conversational_form = ""
            # 生成思维链（CoT）
            cot = virtual_teacher.generate_thinking_chain(
                entry["text"], current_question, entry.get("class", "")
            )

        # 保存结果
        processed_results.append(
            {"conversational_form": conversational_form, "CoT": cot}
        )

    # 添加处理结果到 entry
    entry["virtual_teacher"] = {"processed_results": processed_results}
    # logging.info(f"VirtualTeacher 处理完成: {entry_id}")

    return entry


# 4️⃣ **Simulated Learner**
def process_learner(entry, learner):
    """
    直接处理 SimulatedLearner 逻辑，判断是否使用优化试题或原始试题。
    :param entry: 单条数据条目
    :param learner: SimulatedLearner 实例
    """
    entry_id = entry["id"]  # 使用 entry 中的 ID
    learner_answers = []

    # 获取 question_setter 和 refined_questions
    questions = entry["question_setter"]["questions"]
    refined_questions = entry.get("expert_agent", {}).get("refined_questions", [])

    # 遍历试题，根据索引处理优化试题或原始试题
    for index, question_data in enumerate(questions):
        # 判断是否有优化后的试题
        if index < len(refined_questions) and refined_questions[index].get(
            "requires_refinement", False
        ):
            current_question = refined_questions[index]["refined_response"]
        else:
            current_question = question_data["response"]

        # 生成答案
        learner_answer = learner.answer_question(current_question)
        learner_answers.append({"answer": learner_answer})

    # 添加处理结果到 entry
    entry["simulated_learner"] = {"learner_answers": learner_answers}
    # logging.info(f"SimulatedLearner 处理完成: {entry_id}")

    return entry


# 5️⃣ **Grading Teacher**
def process_grader(entry, grader):
    """
    直接处理 GradingTeacher 逻辑，基于专家的原始试题或优化试题，与学生作答进行评估推理。
    :param entry: 单条数据条目
    :param grader: GradingTeacher 实例
    """
    entry_id = entry["id"]  # 使用 entry 中的 ID
    evaluations = []

    # 获取原始试题和优化试题
    questions = entry["question_setter"]["questions"]
    refined_questions = entry.get("expert_agent", {}).get("refined_questions", [])
    learner_answers = entry["simulated_learner"]["learner_answers"]

    # 遍历每个试题
    for index, question_data in enumerate(questions):
        # 判断是否有优化后的试题
        if index < len(refined_questions) and refined_questions[index].get(
            "requires_refinement", False
        ):
            current_question = refined_questions[index]["refined_response"]
        else:
            current_question = question_data["response"]

        # 获取对应的学生作答
        learner_answer = learner_answers[index]["answer"]

        # 确保有学生作答后再评估
        if learner_answer:
            evaluation = grader.evaluate_answer(
                entry["text"], current_question, learner_answer, entry.get("class", "")
            )
            evaluations.append({"evaluation": evaluation})
        else:
            # 如果没有对应学生作答，记录空评估
            evaluations.append({"evaluation": None})

    # 添加处理结果到 entry
    entry["grading_teacher"] = {"evaluations": evaluations}
    # logging.info(f"GradingTeacher 处理完成: {entry_id}")

    return entry

# ===== tools end ===== #


# ===== main start ===== #
# 包装 process_entry，添加到队列
def process_entry_with_logging(entry, queue: Queue, *args):
    try:
        text_info = entry["text"][:20]
        logging.info(f"Processing entry: {text_info}")
        result = process_entry(entry, *args)  # 调用主处理函数

        # 检查返回值，避免 None 被加入队列
        if result is not None:
            queue.put(result)
        else:
            logging.warning(
                f"跳过空返回值数据: {text_info}"
            )
    except Exception as e:
        logging.error(f"Error processing entry: {text_info}. Details: {e}")


# 加载数据函数省略
def process_entry(entry, out_file, question_setter,
    expert_agent, virtual_teacher, learner,
    grader, step, data_class):
    """
    处理单个数据条目，包括所有阶段，支持阶段性执行，并动态补全前置步骤。
    :param entry: 单条数据条目
    :param out_file: 保存的 JSON 文件路径
    :param question_setter: QuestionSetter 实例
    :param expert_agent: ExpertAgent 实例
    :param virtual_teacher: VirtualTeacher 实例
    :param learner: SimulatedLearner 实例
    :param grader: GradingTeacher 实例
    :param step: 当前执行的步骤 (1-5)
    :data_class: 数据类型 web article book
    """

    # 如果 entry 中没有 ID，则生成并存储
    if "id" not in entry:
        entry["id"] = generate_entry_id(entry)
    entry_id = entry["id"]  # 直接从 entry 中获取 ID

    # 如果 entry 中没有 data_class，则存储
    if "class" not in entry:
        entry["class"] = data_class

    # 文本预处理  📌📌📌  text 在此处被改变
    text_info = entry["text"][:20]
    entry["text"] = preprocess_text(entry)
    if entry["text"] is None:
        # logging.info(f"数据由于低价值或无效而被跳过")
        return None

    # 1️⃣ **检查是否已处理过该条数据**
    existing_entry = find_entry_by_id(out_file, entry_id)
    if existing_entry and existing_entry.get("steps", {}).get(str(step)) == "completed":
        logging.info(f"Step {step}: 已完成，跳过 entry_id={entry_id}")
        return None

    # 合并 entry 数据到 existing_entry 为了把原始语料加进去
    existing_entry.update(entry)

    steps = existing_entry.get("steps", {})

    # 2️⃣ **定义步骤依赖关系**
    step_dependencies = {1: [], 2: [1], 3: [1, 2], 4: [1, 2], 5: [1, 2, 3, 4]}

    # 3️⃣ **逐步检查和执行步骤**
    for required_step in step_dependencies[step]:
        if str(required_step) not in steps or steps[str(required_step)] != "completed":
            if required_step == 1:
                existing_entry = process_question_setter(
                    existing_entry, question_setter
                )
                steps["1"] = "completed"
                logging.info(f"Step 1: QuestionSetter 自动补全完成")

            if required_step == 2:
                existing_entry = process_expert_agent(existing_entry, expert_agent)
                steps["2"] = "completed"
                logging.info(f"Step 2: ExpertAgent 自动补全完成")

            if required_step == 3:
                existing_entry = process_virtual_teacher(
                    existing_entry, virtual_teacher
                )
                steps["3"] = "completed"
                logging.info(f"Step 3: VirtualTeacher 自动补全完成")

            if required_step == 4:
                existing_entry = process_learner(existing_entry, learner)
                steps["4"] = "completed"
                logging.info(f"Step 4: SimulatedLearner 自动补全完成")

    # 4️⃣ **执行目标步骤**
    if str(step) not in steps or steps[str(step)] != "completed":
        if step == 1:
            existing_entry = process_question_setter(existing_entry, question_setter)
        elif step == 2:
            existing_entry = process_expert_agent(existing_entry, expert_agent)
        elif step == 3:
            existing_entry = process_virtual_teacher(existing_entry, virtual_teacher)
        elif step == 4:
            existing_entry = process_learner(existing_entry, learner)
        elif step == 5:
            existing_entry = process_grader(existing_entry, grader)

        steps[str(step)] = "completed"
        logging.info(f"Step {step}: 处理完成: {text_info}")
    else:
        logging.info(f"Step {step}: 已存在，跳过")

    # 5️⃣ **保存数据状态**
    existing_entry["steps"] = steps
    if "text" in existing_entry:
        del existing_entry["text"]  # 删除 text 数据以节省存储空间

    return existing_entry  # 返回处理完成的数据条目


# ===== main end ===== #


# 🔧 **参数解析**
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-path", default="/home/wyp/project/ForestLLM/prompts", help="Prompt文件路径")
    parser.add_argument("--data-file",help="JSONL文件路径，用于加载原始数据")
    parser.add_argument("--out-dir", help="输出文件夹路径，用于保存生成的指令数据集",)
    parser.add_argument("--data_class", default="book", help="数据类别（如 web, article, book）")
    parser.add_argument("--model", default="qwen", choices=["chatgpt_o1-preview", "gpt-4", "chatgpt", "qwen"], type=str)
    parser.add_argument("--num_works", default=1, type=int)
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5], required=True, help="执行阶段",)
    return parser.parse_args()


# 主函数
def main():
    args = parse_args()
    data_file = args.data_file
    out_folder = args.out_dir

    logging.info(f"使用模型: {args.model}")
    logging.info(f"数据文件: {data_file}")
    # logging.info(f"输出路径: {out_file}")
    logging.info(f"当前执行阶段: {args.step}")

    # 自动推断 data_class
    # data_class = infer_data_class(data_file)
    # if data_class == "unknown":
    #     logging.error("无法推断 data_class，请检查数据文件。")
        # data_class = args.data_class

    # 加载已存在的数据
    out_file, existing_ids, existing_data = load_existing_data(
        out_folder, args.model, args.data_class
    )

    # 加载数据和初始化
    data = load_data(data_file)

    # 初始化代理
    question_setter = QuestionSetter(model="qwen")
    expert_agent = ExpertAgent(model="qwen")
    virtual_teacher = VirtualTeacherAgent(model="qwen")
    if args.step >= 4:
        learner = SimulatedLearner(
            model_api=list(args.model),
            model_paths=[
                "/home/wyp/project/swift/models/qwen25_7b_ins",
                "/home/wyp/project/swift/models/minicpm3-4b",
                "/home/wyp/project/swift/models/llama_3_1_8b_ins",
            ],
            model_platforms=["modelscope", "modelscope", "modelscope"],
        )
    else:
        learner = SimulatedLearner(
            model_api=list(args.model),
        )
    grader = GradingTeacher(model="gpt-4")

    # 初始化队列和保存线程
    data_queue = Queue()
    stop_event = Event()
    saver_thread = Thread(target=data_saver, args=(data_queue, out_file, stop_event, args.num_works))
    saver_thread.start()

    # 多线程处理数据
    with ThreadPoolExecutor(max_workers=args.num_works) as executor:  # 根据硬件调整线程数
        futures = {executor.submit(process_entry_with_logging, entry, data_queue, out_file,
                                   question_setter, expert_agent, virtual_teacher,
                                   learner, grader, args.step, args.data_class, ): entry 
                    for entry in data
                    }

        # 使用 tqdm 监控进度
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Entries", unit="entry"):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error in future result: {e}")

    # 等待队列完成所有任务
    data_queue.join()
    stop_event.set()
    saver_thread.join()

    logging.info(f"所有数据已保存到 {out_file}")


if __name__ == "__main__":
    main()


# python tools/run_mutil.py --data-file /home/wyp/project/forest/forestllm-main/mateinfo/all_book.jsonl --out-dir /home/wyp/project/forest/forestllm-main/outputs/0321 --step 3 --data_class book --num_works 8