import json
import re
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 解析 question 和 answer 的函数
def extract_question_and_answer(response):
    """
    从 response 中提取原始 question 和 answer，确保解析准确。
    """
    response = response.strip()

    # 标准匹配（适用于 JSON 格式的 'question' 和 'answer'）
    question_match = re.search(r"[\"']question[\"']\s*[:=]\s*[\"']([^\"']+)[\"']", response, re.IGNORECASE)
    answer_match = re.search(r"[\"']answer[\"']\s*[:=]\s*[\"']([^\"']+)[\"']", response, re.IGNORECASE)

    # 宽松匹配（适用于文本格式或无引号格式）
    if not question_match:
        question_match = re.search(r"question\s*[:=]\s*['\"]?([^'\"]+)", response, re.IGNORECASE)
    if not answer_match:
        answer_match = re.search(r"answer\s*[:=]\s*['\"]?([^'\"]+)", response, re.IGNORECASE)

    # 提取结果
    question = question_match.group(1).strip() if question_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""

    # 记录日志，如果 question 为空
    if not question:
        logging.warning("未能提取到有效的 question，原始 response: %s", response)

    return question, answer

# 文件路径
train_file = "/home/wyp/project/ForestLLM/data_sft/web/train_general_qa.jsonl"
original_data_file = "/home/wyp/project/ForestLLM/qwen_web_output.json"
output_file = "/home/wyp/project/ForestLLM/data_sft/web/train_general_qa_fixed.jsonl"

# 读取原始数据
with open(original_data_file, "r", encoding="utf-8") as f:
    original_data = json.load(f)

# 构建 knowledge 到 response 的映射
knowledge_to_response = {
    item.get("knowledge", "").strip(): item.get("question_setter", {}).get("response", "").strip()
    for item in original_data if item.get("knowledge") and item.get("question_setter", {}).get("response")
}

# 读取训练数据并修复，确保所有数据都被保存
with open(train_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        data = json.loads(line.strip())

        # 检测是否需要修复
        modified = False
        knowledge = data.get("knowledge", "").strip()
        response_text = knowledge_to_response.get(knowledge, "")

        if response_text:
            question, answer = extract_question_and_answer(response_text)

            for msg in data["messages"]:
                if msg["role"] == "user" and not msg["content"].strip():
                    msg["content"] = question
                    modified = True
                elif msg["role"] == "assistant" and not msg["content"].strip():
                    msg["content"] = answer
                    modified = True

        # 无论是否修改，都保存数据
        json.dump(data, f_out, ensure_ascii=False)
        f_out.write("\n")

logging.info(f"修复完成，所有数据已保存至 {output_file}")
