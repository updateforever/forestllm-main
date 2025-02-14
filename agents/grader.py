import json
from agents.agent import BaseAgent
from utils.toolkit import extract_grading_result, run_agent, GradingResult
import re
from pydantic import BaseModel, RootModel, ValidationError
from typing import List


class StudentGrading(BaseModel):
    id: str
    mastery_score: str
    accuracy_score: str
    fluency_score: str


class GradingResultModel(RootModel[List[StudentGrading]]):
    """
    v2 的根模型写法:
    RootModel[内部类型]

    这里的内部类型是 List[List[StudentGrading]]。
    解析后可以通过 .root 来访问。
    """
    pass

class GradingTeacher(BaseAgent):
    """评卷老师Agent，用于全面评估问答数据质量"""

    def __init__(self, model="qwen"):
        super().__init__(name="GradingTeacher", model=model)
        from prompts.finl_eval_prompts import GRADE_PROMPT_CN, GRADE_PROMPT_CN2

        # self.prompt = GRADE_PROMPT_CN
        self.prompt = GRADE_PROMPT_CN2

    def evaluate_answer(self, text, response, student_answer, data_class=None):
        """
        对问答数据进行评估并给出评分和反馈
        - text: 原始背景文本信息
        - question_data: 包含原始试题、专家改进、培训机构专家转换数据等信息的字典
        - student_answer: 虚拟学生的回答
        - data_class: 数据类别（如书籍、文章或网络内容）
        """
        # 创建评估的 prompt
        prompt_template = self.prompt
        prompt = prompt_template.format(
            text=text,
            response=response,
            student_answer=student_answer,
        )

        # 调用大模型进行评估
        grading_response = run_agent(prompt, model=self.model, num_gen=1, temperature=1)

        # 使用正则提取评分和反馈
        try:
            grading_result = self.extract_grading_result(grading_response)
        except json.JSONDecodeError:
            grading_result = {
                "score": "N/A",
                "feedback": "评估生成错误，请检查数据或重新生成。",
            }

        return grading_result

    # def extract_grading_result(self, grading_response):
    #     """从非标准JSON格式的字符串中提取评分和反馈信息"""
    #     """尝试用 Pydantic 模型解析结果"""
    #     try:
    #         # 尝试解析为 JSON
    #         grading_response = re.sub(r"```json", "", grading_response)  # 移除 ```json
    #         grading_response = re.sub(r"```", "", grading_response)  # 移除 ```
    #         grading_response = grading_response.strip()  # 移除首尾空格和换行符
    #         return GradingResult.parse_raw(grading_response).dict()
    #     except ValidationError:
    #         print("Pydantic 无法解析，尝试正则表达式...")
    #         try:
    #             score_match = re.search(
    #                 r"'Score'\s*[:：]\s*['\"]?(\d+)['\"]?", grading_response
    #             )
    #             importance_match = re.search(
    #                 r"[\"']mastery_level[\"']\s*[:：]\s*[\"'](.*?)['\"]",
    #                 grading_response,
    #             )
    #             feedback_match = re.search(r"反馈[:：]\s*(.*?)\n", grading_response)

    #             score = int(score_match.group(1)) if score_match else "N/A"
    #             mastery_level = importance_match.group(1) if importance_match else "N/A"
    #             feedback = feedback_match.group(1) if feedback_match else "N/A"

    #             return {
    #                 "score": score,
    #                 "mastery_level": mastery_level,
    #                 "feedback": feedback,
    #             }
    #         except Exception as e:
    #             print("解析失败:", e)
    #             return {
    #                 "score": "N/A",
    #                 "mastery_level": "N/A",
    #                 "feedback": "无法提取评分和反馈，请检查输入格式。",
    #             }

    def extract_grading_result(self, grading_response: str):
        """
        从非标准JSON格式的字符串中提取评分信息。
        1) 先尝试用 Pydantic 模型 (GradingResultModel) 解析 JSON。
        2) 如果失败，回退到正则表达式或其它方法提取。
        3) 在成功解析后，计算平均 mastery_score 并得到一个总体 mastery_level。
        """
        # 先做一些基本的字符串清理（去除 ```json 等）
        grading_response = re.sub(r"```json", "", grading_response)
        grading_response = re.sub(r"```", "", grading_response)
        grading_response = grading_response.strip()

        # =============== 尝试用 Pydantic 直接解析 ===============
        try:
            # 使用正则匹配 JSON 数组或 JSON 对象
            match = re.search(r'(\[\s*\{.*?\}\s*\]|\{\s*".*?".*?\})', grading_response, re.DOTALL)
            if match:
                grading_response = match.group(1).strip()  # 获取匹配的 JSON 片段

            # 替换 JSON 里的 `null` 为 `"none"`
            grading_response = re.sub(r'\bnull\b', 'none', grading_response)

            # 先把裸露的 `none` 替换成 `"none"`（字符串格式）
            grading_response = re.sub(r'(?<!")\bnone\b(?!")', '"none"', grading_response)

            # 统一所有数值类型（整数和小数）为字符串，例如：5 -> "5"
            grading_response = re.sub(r':\s*(\d+(\.\d+)?)', lambda m: f': "{m.group(1)}"', grading_response)

            # 修正 JSON 格式，去除额外的 `\n`
            grading_response = grading_response.replace("\n", "").strip()

            parsed_model = GradingResultModel.parse_raw(grading_response)
            # 这是 List[StudentGrading]
            items = parsed_model.root

            numeric_scores = []
            for student_item in items:
                ms = student_item.mastery_score  
                if ms.isdigit():
                    numeric_scores.append(int(ms))

            if len(numeric_scores) == 0:
                avg_score = "none"
                mastery_level = "none"
            else:
                avg_val = sum(numeric_scores) / len(numeric_scores)
                avg_score_rounded = round(avg_val)
                # 示范: ≤2 -> l, =3 -> m, ≥4 -> h
                if avg_score_rounded <= 2:
                    mastery_level = "l"
                elif avg_score_rounded == 3:
                    mastery_level = "m"
                else:
                    mastery_level = "h"
                avg_score = int(avg_val)  # 可按需处理成 int(avg_val) 或 round(avg_val,2)

            return {
                "results": [item.dict() for item in items],
                "average_mastery_score": avg_score,
                "mastery_level": mastery_level,
            }

        except ValidationError as e:
            print("Pydantic无法解析，尝试使用正则或其它方式:", e)

            return {
                "results": None,
                "average_mastery_score": "none",
                "mastery_level": "none",
                "message": "解析失败"
            }


def main():
    # 示例测试数据
    text = "这是一个关于生态恢复力的案例研究背景文本。"
    question_data = {
        "response": "根据生态恢复力的概念，恢复力是指系统对外界干扰的适应能力。",
        "expert_feedback": {
            "refined_response": "恢复力是指生态系统对环境变化或干扰的应对和恢复能力。"
        },
        "conversational_form": "请用更通俗的语言描述恢复力。",
        "learner_answer": "恢复力就是生态系统对变化做出反应的能力。",
    }

    data_class = "article"  # 假设数据类别是书籍

    # 初始化 GradingTeacher 实例
    grading_teacher = GradingTeacher(
        prompt_file="/home/wyp/project/ForestLLM/prompts/finl_eval_prompt.json"
    )

    # 调用评估方法
    grading_result = grading_teacher.evaluate_answer(text, question_data, data_class)

    # 打印评估结果
    print("评估结果：")
    print(f"分数: {grading_result['score']}")
    print(f"反馈: {grading_result['feedback']}")


if __name__ == "__main__":
    main()
