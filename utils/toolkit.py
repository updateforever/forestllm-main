import re
from utils.global_methods import *
from pydantic import BaseModel, ValidationError
from typing import Optional


class GradingResult(BaseModel):
    score: int
    # importance_level: str
    # feedback: str
    # importance_level: str = "default_level"  # 提供默认值
    # feedback: str = "No feedback provided"   # 提供默认值
    importance_level: Optional[str] = None  # 允许为空
    feedback: Optional[str] = None          # 允许为空
    mastery_level: str

    
def extract_question(response_data):
    """从 response_data 中提取问题部分"""
    # 直接find
    question_part = response_data[
        response_data.find("question") + 10 : response_data.find("answer") - 1
    ]
    answer = response_data[response_data.find("answer") - 1 :]
    # # 正则化提取
    # question_match = re.search(r"['\"]question['\"]\s*:\s*['\"](\d+)['\"]", response_data)

    # # 正则匹配
    # if question_match:
    #     question_part = question_match.group(1)
    # else:
    #     question_part = ''
    #     print("question not found.{}".format(response_data))

    return question_part, answer


def extract_grading_result(grading_response):
    """从非标准JSON格式的字符串中提取评分和反馈信息"""
    """尝试用 Pydantic 模型解析结果"""
    try:
        # 尝试解析为 JSON
        grading_response = re.sub(r"```json", "", grading_response)  # 移除 ```json
        grading_response = re.sub(r"```", "", grading_response)  # 移除 ```
        grading_response = grading_response.strip()  # 移除首尾空格和换行符
        return GradingResult.parse_raw(grading_response).dict()
    except ValidationError:
        print("Pydantic 无法解析，尝试正则表达式...")
        try:
            score_match = re.search(
                r"'Score'\s*[:：]\s*['\"]?(\d+)['\"]?", grading_response
            )
            importance_match = re.search(
                r"[\"']mastery_level[\"']\s*[:：]\s*[\"'](.*?)['\"]",
                grading_response,
            )
            feedback_match = re.search(r"反馈[:：]\s*(.*?)\n", grading_response)

            score = int(score_match.group(1)) if score_match else "N/A"
            mastery_level = importance_match.group(1) if importance_match else "N/A"
            feedback = feedback_match.group(1) if feedback_match else "N/A"

            return {
                "score": score,
                "mastery_level": mastery_level,
                "feedback": feedback,
            }
        except Exception as e:
            print("解析失败:", e)
            return {
                "score": "N/A",
                "mastery_level": "N/A",
                "feedback": "无法提取评分和反馈，请检查输入格式。",
            }

    # # 使用正则表达式匹配 "score" 和 "feedback"
    # try:
    #     score_match = re.search(
    #         r"'Score'\s*[:：]\s*['\"]?(\d+)['\"]?", grading_response
    #     )
    #     feedback_match = re.search(
    #         r"[\"']importance_level[\"']\s*[:：]\s*[\"'](.*?)[\"']", grading_response
    #     )

    #     score = int(score_match.group(1))
    #     feedback = feedback_match.group(1)
    # except:
    #     score_match = re.search(r"\*\*Score\*\*\s*[:：]\s*(\d+)", grading_response)
    #     feedback = grading_response[
    #         grading_response.find("importance_level")
    #         + 20 : grading_response.find("importance_level")
    #         + 21
    #     ]
    #     score = int(score_match.group(1))

    # # score_match = re.search(r'"score":\s*"(\d+)"', grading_response)
    # # feedback_match = re.search(r'"feedback":\s*"(.*?)"', grading_response, re.DOTALL)

    # # 提取匹配的内容，若匹配不到则使用默认值
    # # score = score_match.group(1) if score_match else "N/A"
    # # feedback = (
    # #     feedback_match.group(1).replace("\\n", " ").replace('\\"', '"')
    # #     if feedback_match
    # #     else "无反馈"
    # # )

    # # 返回结构化的字典
    # return {"score": score, "level": feedback, "feedback": grading_response}


# def run_agent(prompt, model="qwen", num_gen=1, temperature=1):
#     """调用大模型进行生成"""

#     if "qwen" in model:
#         response = run_qwen(prompt, num_gen=num_gen, temperature=temperature)
#     elif "gpt" in model:
#         response = run_chatgpt(
#             prompt, model=model, num_gen=num_gen, temperature=temperature
#         )
#     elif "DeepSeek" in model:
#         response = run_ds(
#             prompt, model=model, num_gen=num_gen, temperature=temperature
#         )
#     else:
#         raise ValueError(f"Unsupported model: {model}")

#     return response


def clean_book_text(text, max_length=2000):
    """
    清洗书籍类数据，移除无关内容（目录、出版社、版权声明等）。
    - text: 原始文本
    - max_length: 最大字符长度
    """
    # 移除包含关键字的行（目录、出版社、版权等）
    patterns_to_remove = [
        r".*目录.*",  # 匹配包含“目录”的行
        r".*Contents.*",  # 匹配包含“Contents”的行
        r".*出版社.*",  # 匹配包含“出版社”的行
        r".*出版时间.*",  # 匹配包含“出版时间”的行
        r".*ISBN.*",  # 匹配包含“ISBN”的行
        r".*版权所有.*",  # 匹配包含“版权所有”的行
        r".*版权声明.*",  # 匹配包含“版权声明”的行
        r".*参考文献.*",  # 匹配包含“参考文献”的行
        r".*附录.*",  # 匹配包含“附录”的行 ["台湾", "毒", "广告", "稿"]
    ]

    # 将匹配的行删除
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # # 尝试定位正文开头
    # start_patterns = [
    #     r"第一章", r"引言", r"前言", r"Chapter 1"
    # ]
    # for pattern in start_patterns:
    #     match = re.search(pattern, text)
    #     if match:
    #         text = text[match.start():]
    #         break

    # 最后截断文本到最大长度
    return text[:max_length]


def extract_sections(text):
    """
    提取文章中的摘要、引言、相关工作和结论部分。
    假定这些部分的关键词为：
    - Abstract / 摘要
    - Introduction / 引言
    - Related Work / 相关工作
    - Conclusion / 结论

    参数：
        text (str): 输入的文章全文。

    返回：
        dict: 包含每个部分内容的字典。
    """
    sections = {
        "abstract": "",
        "introduction": "",
        "related_work": "",
        "conclusion": "",
    }

    # 定义用于匹配各部分的正则表达式
    patterns = {
        "abstract": r"(?:Abstract|摘要)[\s\S]*?(?=Introduction|引言|$)",
        "introduction": r"(?:Introduction|引言)[\s\S]*?(?=Related Work|相关工作|Conclusion|结论|$)",
        "related_work": r"(?:Related Work|相关工作)[\s\S]*?(?=Conclusion|结论|$)",
        "conclusion": r"(?:Conclusion|结论)[\s\S]*?$",
    }

    # 遍历定义的正则模式，匹配对应内容
    for section, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            sections[section] = match.group().strip()

    return sections


# 低价值关键词
LOW_VALUE_KEYWORDS = [
    "林业局",
    "领导讲话",
    "通知公告",
    "简介",
    "人物介绍",
    "官方旗舰店",
    "联系电话",
    "男，",
    "女，",
    "村情概况:",
    "村简介",
    "个人履历:",
    "台湾",
]

# 高价值主题关键词
HIGH_VALUE_KEYWORDS = [
    "生态恢复",
    "植被保护",
    "土壤修复",
    "森林碳储量",
    "水土保持",
    "生物多样性",
    "林业政策",
]


def is_low_value(text):
    """判断文本是否为低价值内容"""
    text = text.lower()
    for keyword in LOW_VALUE_KEYWORDS:
        if keyword in text:
            return True
    return False


def is_high_value(text):
    """判断文本是否为高价值内容"""
    text = text.lower()
    for keyword in HIGH_VALUE_KEYWORDS:
        if keyword in text:
            return True
    return False


def has_low_information_density(text):
    """判断文本信息密度是否低"""
    # 检查文本长度
    if len(text) < 200:
        return True
    # 检查是否有大量非自然语言内容
    if len(re.findall(r"\d+", text)) > 10:  # 大量数字
        return True
    # if len(re.findall(r'<[^>]+>', text)) > 5:  # HTML 标签
    #     return True
    return False


def filter_web_text(entry):
    """
    筛选 web 数据，剔除低质量内容
    """
    text = entry["text"]
    if is_low_value(text):
        return None  # 排除低价值关键词
    if has_low_information_density(text):
        return None  # 排除低信息密度
    # if not is_high_value(text):
    #     return None  # 如果不包含高价值关键词，排除

    # 截断文本长度
    MAX_TEXT_LENGTH = 10000
    return text[:MAX_TEXT_LENGTH]


def parse_output(evaluation_response):
    """
    解析评价输出，支持多种格式（JSON、Markdown 包裹的 JSON、非结构化文本）。

    Args:
        evaluation_response (str): 评价的输出文本。

    Returns:
        dict: 包含 Quality Score、Relevance Score、Consistency Score 和 Improvement Suggestions 的结果。
    """
    result = {
        "quality_score": None,
        "relevance_score": None,
        "consistency_score": None,
        "improvement_suggestions": None,
    }

    # 提取 JSON 部分
    json_match = re.search(r"```json\n(.*?)\n```", evaluation_response, re.S)
    if json_match:
        cleaned_response = json_match.group(1).strip()
    else:
        # 如果没有找到 JSON 部分，清理 Markdown 格式标记（```json）以及多余的换行符和空格
        cleaned_response = re.sub(r"```json\n?|```", "", evaluation_response).strip()

    try:
        # 尝试直接解析 JSON 格式
        evaluation_data = json.loads(cleaned_response)
        result["quality_score"] = int(evaluation_data.get("Quality Score", 0))
        result["relevance_score"] = int(evaluation_data.get("Relevance Score", 0))
        result["consistency_score"] = int(evaluation_data.get("Consistency Score", 0))
        result["improvement_suggestions"] = evaluation_data.get(
            "Improvement Suggestions", ""
        ).strip()
        return result
    except (json.JSONDecodeError, ValueError):
        # 如果 JSON 解析失败，使用正则表达式提取分数
        try:
            quality_score_match = re.search(
                r"'Quality Score'\s*[:：]\s*['\"]?(\d+)['\"]?", evaluation_response
            )
            relevance_score_match = re.search(
                r"'Relevance Score'\s*[:：]\s*['\"]?(\d+)['\"]?", evaluation_response
            )
            consistency_score_match = re.search(
                r"'Consistency Score'\s*[:：]\s*['\"]?(\d+)['\"]?", evaluation_response
            )

            if quality_score_match:
                result["quality_score"] = int(quality_score_match.group(1))
            if relevance_score_match:
                result["relevance_score"] = int(relevance_score_match.group(1))
            if consistency_score_match:
                result["consistency_score"] = int(consistency_score_match.group(1))
        except Exception as e:
            print(f"正则解析分数失败: {e}")

    # # 如果正则表达式也失败，使用 find 查找改进建议
    # if not result["improvement_suggestions"]:
    #     try:
    #         improvement_start = evaluation_response.find("Improvement Suggestions")
    #         if improvement_start != -1:
    #             result["improvement_suggestions"] = evaluation_response[
    #                 improvement_start + 25 :
    #             ].strip()
    #     except Exception as e:
    #         print(f"find 方法提取改进建议失败: {e}")

    return result


'''
def initialize_agents(prompt_path, model="qwen"):
    """初始化各个Agent"""
    question_prompt_file = os.path.join(prompt_path, "question_prompts.json")
    question_setter = QuestionSetter(prompt_file=question_prompt_file, model=model)
    expert_prompt_file = os.path.join(prompt_path, "expert_prompts.json")
    expert_agent = ExpertAgent(prompt_file=expert_prompt_file, model=model)
    institute_prompt_file = os.path.join(prompt_path, "traininginstitute_prompts.json")
    training_expert = VirtualTeacherAgent(
        prompt_file=institute_prompt_file, model=model
    )
    student_prompt_file = os.path.join(prompt_path, "student_prompts.json")
    learner = SimulatedLearner(
        prompt_file=student_prompt_file,
        model_api=["qwen"],
        model_paths=["/home/wyp/project/swift/models/qwen25_7b_ins"],
        model_platforms=['modelscope']
        )

    grading_prompt_file = os.path.join(prompt_path, "finl_eval_prompt.json")
    grader = GradingTeacher(prompt_file=grading_prompt_file, model=model)
    return question_setter, expert_agent, training_expert, learner, grader
'''
