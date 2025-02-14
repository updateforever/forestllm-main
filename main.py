"""
Author: updateforever 1456742985@qq.com
Date: 2024-12-03 16:09:06
LastEditors: updateforever 1456742985@qq.com
LastEditTime: 2024-12-05 17:37:40
FilePath: \ForestLLM\main.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""

import hashlib
import logging
import re
import argparse
import os
import json
from agents import (
    QuestionSetter,
    ExpertAgent,
    VirtualTeacherAgent,
    SimulatedLearner,
    GradingTeacher,
)

from utils.toolkit import clean_book_text, filter_web_text

# 设置日志记录
logging.basicConfig(level=logging.INFO)


def load_data(data_file):
    """从JSONL文件中加载数据"""
    data = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    logging.info(f"加载了 {len(data)} 条数据")
    return data


def load_existing_data(out_folder, model_name, data_class):
    """加载已有的生成数据"""
    # 根据模型名称和文件夹路径生成对应的输出文件路径
    out_file = os.path.join(out_folder, f"{model_name}_{data_class}_output.json")

    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            logging.info(
                f"加载了 {len(existing_data)} 条已生成数据，模型: {model_name}"
            )
            return out_file, {entry["id"] for entry in existing_data}, existing_data
    else:
        logging.warning(f"未找到模型 {model_name} 对应的输出文件: {out_file}")
        return out_file, set(), []


def generate_entry_id(entry):
    """根据数据内容生成唯一的ID"""
    # 将数据条目转换为JSON字符串
    entry_str = json.dumps(entry, sort_keys=True, ensure_ascii=False)
    # 使用SHA256对数据进行哈希，确保相同内容生成相同ID
    return hashlib.sha256(entry_str.encode("utf-8")).hexdigest()


def initialize_agents(model="qwen"):
    # 初始化
    question_setter = QuestionSetter(model="gpt-4")

    expert_agent = ExpertAgent(model="qwen")

    training_expert = VirtualTeacherAgent(model=model)

    learner = SimulatedLearner(
        model_api=list(model),
        model_paths=[
            "/home/wyp/project/swift/models/qwen25_7b_ins",
            "/home/wyp/project/swift/models/minicpm3-4b",
            "/home/wyp/project/swift/models/llama_3_1_8b_ins",
        ],
        model_platforms=["modelscope", "modelscope", "modelscope"],
    )

    grader = GradingTeacher(model="gpt-4")

    return question_setter, expert_agent, training_expert, learner, grader


def generate_instruction_data(
    data,
    existing_ids,
    question_setter,
    expert_agent,
    training_expert,
    learner,
    grader,
    out_file,
    debug=True,
):
    """通过各个Agent协作生成指令数据，并通过反馈循环优化"""
    for entry in data:
        # 为当前数据条目生成唯一id
        entry_id = generate_entry_id(entry)
        if entry_id in existing_ids:
            logging.info(f"跳过已处理的数据: {entry_id}")
            continue

        text = entry["text"]
        data_class = (
            entry.get("class", "")
            if entry.get("class", None)
            else entry.get("meta_info").get("data_class", None)
        )
        session_data = {
            "id": entry_id,
            "data_class": data_class,
            "questions": [],
        }  # 为整条数据赋予唯一id

        if data_class == "article":
            MAX_TEXT_LENGTH = 10000
            text = text[:MAX_TEXT_LENGTH]
        elif data_class == "web":
            # 数据筛选函数
            text = filter_web_text(entry)
            if text is None:
                logging.info(f"跳过低价值 Web 数据: {entry_id}")
                continue
        elif data_class == "book":
            MAX_TEXT_LENGTH = 10000
            text = clean_book_text(text, max_length=MAX_TEXT_LENGTH)

        #     # 如果是文章，提取段落
        #     sections = extract_sections(text)
        #     # 整合段落为一个完整的文本字符串
        #     text = "\n\n".join(
        #         section_text
        #         for section_text in [
        #             sections.get("abstract", ""),
        #             sections.get("introduction", ""),
        #             sections.get("related_work", ""),
        #             sections.get("conclusion", ""),
        #         ]
        #         if section_text  # 排除空部分
        #     )

        logging.info(f"开始生成会话，数据类别为: {data_class}")

        if debug:
            # Step 1: 出题人生成多种题型的初始问题和答案
            questions = question_setter.generate_response(text, data_class)

            for question_data in questions:
                # Step 2: 专家Agent扩展问题和答案
                expert_feedback = expert_agent.evaluate_and_refine_question(
                    text, question_data, data_class
                )
                question_data["expert_feedback"] = expert_feedback

                # Step 3: 培训机构专家转换为对话格式
                # dialogue_data = training_expert.convert_to_conversational_form(
                #     text, question_data, data_class
                # )

                # Step 3.1: 生成思维链
                cot_chain = training_expert.generate_thinking_chain(
                    text, question_data, data_class
                )
                question_data["CoT"] = cot_chain
                # Step 3.2: 判断是否为选择题，进行口语化转换
                if question_data.get("question_type") == "multiple_choice":
                    dialogue_data = training_expert.convert_to_conversational_form(
                        text, question_data, data_class
                    )
                    question_data["conversational_form"] = dialogue_data
                else:
                    question_data["conversational_form"] = ""

                # Step 4: 模拟考生回答问题
                learner_answer = learner.answer_question(question_data)
                question_data["learner_answer"] = learner_answer

                # Step 5: 评卷老师比对答案并评估
                evaluation = grader.evaluate_answer(text, question_data, data_class)
                question_data["evaluation"] = evaluation

                # 保存当前问题数据
                session_data["questions"].append(question_data)

            # 保存当前条目到文件
            save_partial_data(out_file, session_data)
        else:
            try:
                # Step 1: 出题人生成多种题型的初始问题和答案
                questions = question_setter.generate_response(text, data_class)

                for question_data in questions:
                    # Step 2: 专家Agent扩展问题和答案
                    expert_feedback = expert_agent.evaluate_and_refine_question(
                        text, question_data, data_class
                    )
                    question_data["expert_feedback"] = expert_feedback

                    # Step 3: 培训机构专家转换为对话格式
                    dialogue_data = training_expert.convert_to_conversational_form(
                        text, question_data, data_class
                    )
                    question_data["conversational_form"] = dialogue_data

                    # Step 4: 模拟考生回答问题
                    learner_answer = learner.answer_question(text, question_data)
                    question_data["learner_answer"] = learner_answer

                    # Step 5: 评卷老师比对答案并评估
                    evaluation = grader.evaluate_answer(text, question_data, data_class)
                    question_data["evaluation"] = evaluation

                    # 保存当前问题数据
                    session_data["questions"].append(question_data)

                # 保存当前条目到文件
                save_partial_data(out_file, session_data)

            except Exception as e:
                logging.error(f"处理数据 {entry_id} 时发生错误: {e}")
                continue


def save_partial_data(out_file, session_data):
    """保存生成的单条数据到文件"""
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    # 如果文件存在，追加写入；否则创建文件
    if os.path.exists(out_file):
        with open(out_file, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data.append(session_data)
            f.seek(0)
            json.dump(data, f, indent=4, ensure_ascii=False)
    else:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump([session_data], f, indent=4, ensure_ascii=False)

    logging.info(f"已保存数据 {session_data['id']} 到 {out_file}")


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
        help="输出文件夹路径，用于保存生成的指令数据集",
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
    args = parser.parse_args()
    return args


def main():

    # get arguments
    args = parse_args()

    print("******************  Evaluating Model %s ***************" % args.model)

    # 加载数据和已有生成数据
    data = load_data(args.data_file)
    out_file, existing_ids, _ = load_existing_data(
        args.out_dir, args.data_class, args.model
    )

    # 初始化所有Agent
    question_setter, expert_agent, training_expert, learner, grader = initialize_agents(
        args.model
    )

    # 生成指令数据（直接即时保存，无需返回）
    generate_instruction_data(
        data,
        existing_ids,
        question_setter,
        expert_agent,
        training_expert,
        learner,
        grader,
        out_file,
    )


if __name__ == "__main__":
    main()
