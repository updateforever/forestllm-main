import os
import json
import hashlib
import logging
import argparse
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


# 配置 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process.log", mode='w', encoding='utf-8'),  # 输出到文件
        # logging.StreamHandler()  # 如果不需要在终端显示日志，可以注释掉这行
    ]
)


# 替换 logging 的输出方式
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        # 将日志记录输出到 tqdm 的 write 方法中
        msg = self.format(record)
        tqdm.write(msg)

# 添加自定义的 TqdmLoggingHandler
# logging.getLogger().addHandler(TqdmLoggingHandler())

def process_entry_with_logging(entry, *args):
    """
    包装 process_entry 函数，添加日志记录。
    """
    try:
        text_info = entry["text"][:20]
        logging.info(f"Processing entry: {text_info}")
        process_entry(entry, *args)
        return f"Completed: {text_info}"
    except Exception as e:
        logging.error(f"Error processing entry: {entry['text'][:20]}. Details: {e}")
        return f"Failed: {entry['text'][:20]}"
    

# 生成唯一 ID
def generate_entry_id(entry):
    entry_str = json.dumps(entry, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(entry_str.encode("utf-8")).hexdigest()


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
            data_class = first_entry.get("class", "unknown")
            logging.info(f"自动推断到 data_class: {data_class}")
            return data_class
    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        logging.error(f"推断 data_class 时出错: {e}")
        return "unknown"


def load_existing_data(out_folder, model_name, data_class):
    """
    加载已存在的模型生成数据（支持 JSON 和 JSONL 格式）
    :param out_folder: 输出文件夹路径
    :param model_name: 模型名称
    :param data_class: 数据类别（如 web, article, book）
    :return: (out_file, id_set, existing_data)
    """
    out_file = os.path.join(out_folder, f"{model_name}_{data_class}_output.json")

    existing_data = []
    id_set = set()

    if os.path.exists(out_file):
        logging.info(f"检测到文件: {out_file}，开始加载已生成数据...")
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for entry in data:
                    if "id" in entry:
                        id_set.add(entry["id"])
                        existing_data.append(entry)
        except json.JSONDecodeError:
            logging.error(f"文件解析失败: {out_file}")
        except Exception as e:
            logging.error(f"加载文件时发生错误: {e}")
    else:
        logging.warning(f"未找到模型 {model_name} 对应的输出文件: {out_file}")

    logging.info(f"成功加载 {len(existing_data)} 条数据，来自文件: {out_file}")
    return out_file, id_set, existing_data


# 保存数据到 JSON 文件
def save_data(data_file, data):
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


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

    try:
        # 检查文件扩展名，判断是 JSON 还是 JSONL 格式
        if out_file.endswith(".json"):
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for entry in data:
                    if entry.get("id") == entry_id:
                        return entry

        elif out_file.endswith(".jsonl"):
            with open(out_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("id") == entry_id:
                            return entry
                    except json.JSONDecodeError:
                        logging.warning(f"跳过无效的 JSON 行: {line.strip()}")
                        continue

        else:
            logging.error(f"不支持的文件格式: {out_file}")
            return {"id": entry_id, "steps": {}}

    except json.JSONDecodeError as e:
        logging.error(f"JSON 解析失败: {e}")
    except Exception as e:
        logging.error(f"查找 ID 时发生未知错误: {e}")

    return {"id": entry_id, "steps": {}}


# 检查前置数据是否存在
def check_required_keys(entry, required_keys):
    return all(key in entry and entry[key] for key in required_keys)


def preprocess_text(entry):
    """
    根据数据类型（data_class）对文本进行预处理
    """
    data_class = entry.get("class", "")
    text = entry.get("text", "")

    if data_class == "article":
        MAX_TEXT_LENGTH = 10000
        text = text[:MAX_TEXT_LENGTH]
        # logging.info(f"文章类数据已截断到 {MAX_TEXT_LENGTH} 字符")

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


# **Entry Processing Pipeline**
def process_entry(
    entry,
    out_file,
    question_setter,
    expert_agent,
    virtual_teacher,
    learner,
    grader,
    step,
    data_class,
):
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
        logging.info(f"数据由于低价值或无效而被跳过")
        return

    # 1️⃣ **加载数据状态**
    existing_entry = find_entry_by_id(out_file, entry_id)
    if not existing_entry:
        existing_entry = {"id": entry_id, "steps": {}}

    # 合并 entry 数据到 existing_entry
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
                existing_entry = process_virtual_teacher(existing_entry, virtual_teacher)
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

    if os.path.exists(out_file):
        with open(out_file, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data = [e for e in data if e.get("id") != entry_id]
            data.append(existing_entry)
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=4)
    else:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump([existing_entry], f, ensure_ascii=False, indent=4)

    logging.info(f"数据状态已保存")


# 🔧 **参数解析**
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-path",
        help="Prompt文件路径，用于加载出题人、专家、培训机构专家、模拟考生和评卷老师的Prompt",
        default="/home/wyp/project/ForestLLM/prompts",
    )
    parser.add_argument(
        "--data-file",
        help="JSONL文件路径，用于加载原始数据",
    )
    parser.add_argument(
        "--out-dir",
        help="输出文件夹路径，用于保存生成的指令数据集",
    )
    parser.add_argument(
        "--data_class",
        default="web",
        help="数据类别（如 web, article, book）",
    )
    parser.add_argument(
        "--model",
        default="qwen",
        choices=[
            "chatgpt_o1-preview",
            "gpt-4",
            "chatgpt",
            "claude",
            "gemini",
            "qwen",
            "gpt-3.5-turbo",
        ],
        type=str,
    )
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3, 4, 5],
        required=True,
        help="选择要执行的阶段: 1=QuestionSetter, 2=ExpertAgent, 3=VirtualTeacher, 4=SimulatedLearner, 5=GradingTeacher",
    )
    return parser.parse_args()


# 📝 **主函数**
def main():
    args = parse_args()
    data_file = args.data_file
    out_folder = args.out_dir

    # 动态生成输出文件路径
    out_file = os.path.join(out_folder, f"{args.model}_{args.data_class}_output.json")

    logging.info(f"使用模型: {args.model}")
    logging.info(f"数据文件: {data_file}")
    logging.info(f"输出路径: {out_file}")
    logging.info(f"当前执行阶段: {args.step}")

    # 自动推断 data_class
    data_class = infer_data_class(data_file)
    if data_class == "unknown":
        logging.error("无法推断 data_class，请检查数据文件。")
        return

    # 加载已存在的数据
    out_file, existing_ids, existing_data = load_existing_data(
        out_folder, args.model, data_class
    )

    # 加载原始数据
    data = load_data(data_file)

    # 初始化 Agents
    question_setter = QuestionSetter(model=args.model)
    expert_agent = ExpertAgent(model=args.model)
    virtual_teacher = VirtualTeacherAgent(model=args.model)
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

     # 使用 ThreadPoolExecutor 并发处理数据
    with ThreadPoolExecutor(max_workers=24) as executor:  # 设置最大线程数，可根据硬件配置调整
        futures = {
            executor.submit(
                process_entry_with_logging,
                entry,
                out_file,
                question_setter,
                expert_agent,
                virtual_teacher,
                learner,
                grader,
                args.step,
                data_class,
            ): entry
            for entry in data
        }

        # 使用 tqdm 监控进度条
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Entries", unit="entry"):
            try:
                result = future.result()
                logging.info(result)
            except Exception as e:
                logging.error(f"Error in future result: {e}")

    # for entry in tqdm(data, desc="Processing Entries", unit="entry"):
    #     try:
    #         text_info = entry["text"][:20]
    #         logging.info(f"Processing entry: {text_info}")
    #         process_entry(
    #             entry,
    #             out_file,
    #             question_setter,
    #             expert_agent,
    #             virtual_teacher,
    #             learner,
    #             grader,
    #             args.step,
    #             data_class,
    #         )
    #     except Exception as e:
    #         logging.error(f"Error processing entry: {text_info}, skipping. Details: {e}")

    # # 逐条处理数据，根据 step 控制执行阶段
    # for entry in data:
    #     text_info = entry["text"][:20]
    #     logging.info(f"Processing entry: {text_info}")
    #     process_entry(
    #         entry,
    #         out_file,
    #         question_setter,
    #         expert_agent,
    #         virtual_teacher,
    #         learner,
    #         grader,
    #         args.step,
    #         data_class,
    #     )


    logging.info(f"所有数据已保存到 {out_file}")





if __name__ == "__main__":
    main()
