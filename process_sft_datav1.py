import json
import pandas as pd
import argparse
import re
from agents import VirtualTeacherAgent

# 初始化 VirtualTeacherAgent
virtual_teacher_r1 = VirtualTeacherAgent(model="deepseek-r1")
virtual_teacher = VirtualTeacherAgent(model="qwen")


def extract_conversational_form(item, log_file="conversation_errors.log"):
    """
    提取选择题的口语化对话
    - 如果解析失败，记录 `id` 和 `conversational_form` 到日志文件
    """
    response = item.get("virtual_teacher", {}).get("conversational_form", "{}")

    response = re.sub(r"```json", "", response)  # 移除 ```json
    response = re.sub(r"```", "", response)  # 移除 ```
    response = response.strip()

    try:
        conversational_data = json.loads(response)
        return conversational_data.get("input", ""), conversational_data.get(
            "output", ""
        )
    except json.JSONDecodeError:
        # 解析失败，增加错误计数
        knowledge_point = item.get("question_setter", {}).get("knowledge", "未知知识点")
        response_text = item.get("question_setter", {}).get("response", "未知问题")
        # 调用 VirtualTeacherAgent 进行修正
        try:
            conversational_response = virtual_teacher.convert_to_conversational_form(
                text='', response=response, data_class="web"
            )
            conversational_data = json.loads(conversational_response)
            return conversational_data.get("input", ""), conversational_data.get(
                "output", ""
            )
        except Exception as e:
            # 退而求其次，尝试手动解析 input/output
            input_start = conversational_response.find("input") + 9
            input_end = conversational_response.find(",", input_start) - 2
            output_start = conversational_response.find("output") + 10

            if input_start > 8 and input_end > input_start and output_start > 9:
                return conversational_response[input_start:input_end], conversational_response[output_start:]
            else:
                with open(log_file, "a", encoding="utf-8") as log_f:
                    log_f.write(
                        json.dumps(
                            {"id": item.get("id", "unknown_id"), "knowledge": knowledge_point, "question": response_text,
                            "error": str(e)}, ensure_ascii=False
                        )
                    )
                    log_f.write("\n")
                return "自动修正失败", "自动修正失败"  # 返回默认值

def extract_question_and_answer(response):
    """
    解析 response，提取 question 和 answer，确保解析准确，支持多行 answer。
    """
    response = response.strip()

    # 去除 json 格式标注
    response = re.sub(r"```json", "", response)
    response = re.sub(r"```", "", response)
    response = response.strip()

    # 解析 question
    question_match = re.search(r"[\"']question[\"']\s*[:=]\s*[\"']([^\"']+)[\"']", response, re.IGNORECASE)

    # 解析 answer，支持 ''' 多行、"" 多行、单行
    answer_match = re.search(r"[\"']answer[\"']\s*[:=]\s*['\"]{3}([\s\S]*?)['\"]{3}", response, re.IGNORECASE | re.DOTALL)

    # 如果标准匹配失败，尝试匹配无 `'''` 或 `"""` 但带换行符的 answer
    if not answer_match:
        answer_match = re.search(r"[\"']answer[\"']\s*[:=]\s*['\"]([\s\S]*?)['\"]", response, re.IGNORECASE | re.DOTALL)

    # 宽松匹配（适用于文本格式或无引号格式）
    if not question_match:
        question_match = re.search(r"question\s*[:=]\s*['\"]?([^'\"]+)", response, re.IGNORECASE)
    if not answer_match:
        answer_match = re.search(r"answer\s*[:=]\s*['\"]?([^'\"]+)", response, re.IGNORECASE)

    # 提取结果
    question = question_match.group(1).strip() if question_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""

    # **4. 如果正则匹配失败，使用 `find()` 方法**
    if not answer:
        answer_start = -1
        for keyword in ["**答案**", "**answer**", "答案", "answer", "'答案'", "'answer'", '"答案"', '"answer"']:
            answer_start = response.find(keyword)
            if answer_start != -1:
                break

        if answer_start != -1:
            answer_start += len(keyword)  # 移动到答案内容
            answer = response[answer_start:].strip()
            
            # **去掉前导 `:` 或 `=`，防止污染 answer**
            if answer.startswith(":"):
                answer = answer[1:].strip()
            elif answer.startswith("="):
                answer = answer[1:].strip()

    if not question:
        question_start = -1
        for keyword in ["简答题", "**问题**", "**question**", "问题", "question", "'问题'", "'question'", '"问题"', '"question"']:
            question_start = response.find(keyword)
            if question_start != -1:
                break

        if question_start != -1:
            question_start += len(keyword)  # 移动到问题内容
            question = response[question_start:].strip()

            # **确保 question 不会包含 answer**
            answer_pos = min(
                [response.find(k, question_start) for k in 
                ["**答案**", "**answer**", "答案", "answer", "'答案'", "'answer'", '"答案"', '"answer"']
                if response.find(k, question_start) != -1],  # 只保留找到的索引
                default=float("inf")  # 设定默认值，防止所有都未找到时报错
            )
            if answer_pos != float("inf"):
                question = response[question_start:answer_pos].strip()
                
        # **去掉前导 `:` 或 `=`，防止污染 question**
        if question.startswith(":"):
            question = question[1:].strip()
        elif question.startswith("="):
            question = question[1:].strip()

    if not question or not answer:
        print(f"解析失败: {response}...")  # 打印字符


    return question, answer


def extract_cot_answer(item, mastery_level):
    """
    处理 JSONL 里简答题 & 开放讨论题的 CoT 逻辑：
    - 高掌握度 (`mastery_level == "h"`) → 直接存储答案
    - 低掌握度 (`mastery_level != "l" 和 "m"`) → 在答案前加 `<think>CoT内容</think>`
    """
    response_text = item.get("question_setter", {}).get("response", "")
    question, answer = extract_question_and_answer(response_text)
    cot = item.get("virtual_teacher", {}).get("CoT", "")

    if cot == "":
        # 调用
        response = virtual_teacher_r1.cot_deepseek(response=question)
        cot = response["reasoning"]
        answer = response["answer"]
    
    if mastery_level in ["l", "m"] and cot:
        answer = f"<think>{cot}</think> {answer}"

    return question, answer


# def extract_multiple_choice_details(response):
#     """
#     提取多选题的选项、正确答案和解释
#     """
#     response = response.strip()

#     # 1️⃣ 提取 `question`（去掉选项部分）
#     question_match = re.search(r"'question':\s*'(.*?)'", response, re.DOTALL)
#     if not question_match:
#         return "", ["", "", "", ""], "", ""

#     question_value_escaped = question_match.group(1)

#     # **分割行，提取问题部分**
#     question_value_escaped = question_value_escaped.replace("\\n", "\n")
#     question_lines = question_value_escaped.split("\n")
#     question = question_lines[0].strip()  # 取第一行作为问题
#     options_text = "\n".join(question_lines[1:])  # 其余部分是选项

#     # 2️⃣ 提取选项（A, B, C, D）
#     option_matches = re.findall(r"([A-D])\.\s*([^\n]+)", options_text)
#     option_dict = {opt[0]: opt[1].strip() for opt in option_matches}

#     # **确保 A-D 选项完整**
#     choices = ["A", "B", "C", "D"]
#     option_values = [option_dict.get(choice, "") for choice in choices]

#     # 3️⃣ 提取正确答案
#     answer_match = re.search(r"'answer':\s*'([A-D])'", response)
#     answer = answer_match.group(1) if answer_match else ""

#     # 4️⃣ 提取解释（如果有）
#     explanation_match = re.search(r"'explanation':\s*'([^']*)'", response)
#     explanation = explanation_match.group(1) if explanation_match else ""

#     return question, option_values, answer, explanation


def extract_multiple_choice_details(response):
    """
    提取多选题的选项、正确答案和解释，并确保数据完整。
    """
    response = response.strip()

    # **1️⃣ 提取 `question`（去掉选项部分）**
    question_match = re.search(r"'question':\s*'(.*?)'", response, re.DOTALL)
    if not question_match:
        print("⚠️ 未找到 question")
        return "", ["", "", "", ""], "", ""

    question_value_escaped = question_match.group(1).replace("\\n", "\n")
    question_lines = question_value_escaped.split("\n")

    # **检查 question 是否有效**
    question = question_lines[0].strip() if question_lines else "数据缺失"
    if not question:
        print("⚠️ question 为空")
        question = "数据缺失"

    # **2️⃣ 提取选项（A, B, C, D）**
    options_text = "\n".join(question_lines[1:])
    option_matches = re.findall(r"([A-D])\.\s*([^\n]+)", options_text)
    option_dict = {opt[0]: opt[1].strip() for opt in option_matches}

    choices = ["A", "B", "C", "D"]
    option_values = [option_dict.get(choice, "数据缺失") for choice in choices]

    # **检查选项是否为空**
    for i, opt in enumerate(option_values):
        if not opt:
            print(f"⚠️ 选项 {choices[i]} 为空")
            option_values[i] = "数据缺失"

    # **3️⃣ 提取正确答案**
    answer_match = re.search(r"'answer':\s*'([A-D])'", response)
    answer = answer_match.group(1) if answer_match else "数据缺失"
    if answer == "数据缺失":
        print("⚠️ 未找到 answer")

    # **4️⃣ 提取解释（如果有）**
    explanation_match = re.search(r"'explanation':\s*'([^']*)'", response)
    explanation = explanation_match.group(1) if explanation_match else "数据缺失"
    if not explanation:
        print("⚠️ explanation 为空")
        explanation = "数据缺失"

    return question, option_values, answer, explanation


def process_data(input_file, train_output_files, eval_output_files):
    """
    读取数据，并分类转换为 JSONL & CSV 格式，同时保留 `id`，确保后续能基于 `id` 进行数据划分。
    现在 `eval_multiple_choice.csv` 由 `pandas` 处理，自动处理逗号问题。
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data_samples = json.load(f)

    multiple_choice_id = 1  # 评测序号（CEval）
    train_streams = {key: open(path, 'w', encoding='utf-8') for key, path in train_output_files.items()}
    eval_streams = {key: open(path, 'w', encoding='utf-8') for key, path in eval_output_files.items()}

    eval_csv_data = []

    for item in data_samples:
        item_id = item.get("id", "unknown_id")
        mastery_level = item.get("grading_teacher", {}).get("evaluation", {}).get("mastery_level", "unknown")
        question_type = item.get("question_setter", {}).get("question_type", "unknown")
        knowledge_point = item.get("question_setter", {}).get("knowledge", "")
        response_text = item.get("question_setter", {}).get("response", "")

        if question_type == "multiple_choice":
            user_input, assistant_output = extract_conversational_form(item)
            messages = [
                {"role": "system", "content": "你是一个专业的林业智能问答助手"},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_output},
            ]
            json.dump(
                {
                    "id": item_id,
                    "messages": messages,
                    "mastery_level": mastery_level,
                    "question_type": question_type,
                    "knowledge": knowledge_point,
                },
                train_streams["multiple_choice"],
                ensure_ascii=False
            )
            train_streams["multiple_choice"].write('\n')

            # 评测 CSV（存储原始试题）
            question, choices, answer, explanation = extract_multiple_choice_details(response_text)
            if not question or not any(choices) or not answer:
                print(f"⚠️ 跳过无效题目（id: {item_id}，question、choices 或 answer 为空）")
                continue
            eval_csv_data.append([multiple_choice_id, question] + choices + [answer, explanation, item_id, knowledge_point])
            multiple_choice_id += 1

        elif question_type in ["short_answer", "open_discussion"]:
            question, answer = extract_cot_answer(item, mastery_level)
            messages = [
                {"role": "system", "content": "你是一个专业的林业智能问答助手"},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]

            json.dump(
                {
                    "id": item_id,
                    "messages": messages,
                    "mastery_level": mastery_level,
                    "question_type": question_type,
                    "knowledge": knowledge_point,
                },
                train_streams["general_qa"],
                ensure_ascii=False
            )
            train_streams["general_qa"].write('\n')

            question, answer = extract_question_and_answer(response_text)
            eval_qa_entry = {"id": item_id, "history": [], "query": question, "response": answer}
            json.dump(eval_qa_entry, eval_streams["general_qa"], ensure_ascii=False)
            eval_streams["general_qa"].write('\n')

    for f in train_streams.values():
        f.close()
    for f in eval_streams.values():
        f.close()

    # ✅ **使用 pandas 存储 CSV（自动处理逗号和格式问题）**
    eval_df = pd.DataFrame(eval_csv_data, columns=["eval_id", "question", "A", "B", "C", "D", "answer", "explanation", "id", "knowledge_point"])
    eval_df.to_csv(eval_output_files["multiple_choice"], encoding="utf-8", index=False)

    print(f"数据已成功分类并保存（所有数据均保留 `id`）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data for training and evaluation.")
    parser.add_argument("--input", type=str, default="qwen_web_output.json", help="Path to the input JSON file")
    parser.add_argument("--train_mc", type=str, default="outputs/sft_data/web/train_multiple_choice.jsonl")
    parser.add_argument("--train_qa", type=str, default="outputs/sft_data/web/train_general_qa.jsonl")
    parser.add_argument("--eval_mc", type=str, default="outputs/sft_data/web/eval_multiple_choice.csv")
    parser.add_argument("--eval_qa", type=str, default="outputs/sft_data/web/eval_general_qa.jsonl")

    args = parser.parse_args()

    process_data(args.input, {"multiple_choice": args.train_mc, "general_qa": args.train_qa},
                    {"multiple_choice": args.eval_mc, "general_qa": args.eval_qa})
