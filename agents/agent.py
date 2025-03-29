import json
from utils.global_methods import *
import re
from utils.toolkit import *
from prompts import *


class BaseAgent:
    """Agent基础类，为所有子Agent提供通用接口和基本功能"""

    def __init__(self, name="BaseAgent", model="qwen"):
        self.name = name
        self.feedback_history = []  # 用于存储评估反馈
        self.model = model

    def load_prompts(self, prompt_file):
        """加载 prompt 文件"""
        with open(prompt_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def receive_feedback(self, feedback):
        """接受评估反馈的通用方法"""
        self.feedback_history.append(feedback)
        print(f"{self.name} 收到反馈: {feedback}")

    def generate_response(self, input_data):
        """生成响应的通用方法，具体逻辑在子类中实现"""
        raise NotImplementedError("子类必须实现该方法")


class QuestionSetter(BaseAgent):
    """出题人Agent，生成各类题型的问题和标准答案"""

    def __init__(self, model="qwen"):
        super().__init__(name="QuestionSetter", model=model)
        from prompts.question_prompts import QUESTION_PROMPTS_CN

        self.prompts = QUESTION_PROMPTS_CN

    def generate_response(self, input_data, data_class):
        """生成完整的试卷结构"""
        # 步骤1: 从输入中罗列知识点并进行复杂度分析
        knowledge_points = self.extract_knowledge_points(input_data, data_class)

        # 步骤2: 根据知识点的难度分配题型
        question_set = []
        for point, difficulty, original_question in knowledge_points:
            question_set.extend(
                self.generate_questions_for_point(point, original_question, difficulty, input_data)
            )

        return question_set

    def extract_knowledge_points(self, text, data_class):
        """根据数据类别从文本中提取知识点并评估难度"""
        prompt_template = self.prompts[f"knowledge_extraction_{data_class}"]

        # 构建 prompt
        prompt = prompt_template.format(text=text, length=len(text), q_num=len(text)/500)

        # 调用大模型生成知识点
        response = run_agent(prompt, model=self.model, num_gen=1, temperature=1)

        # 使用正则表达式去除格式标记，如 ```json 和 ```
        json_content = re.sub(r"```(?:json)?|```", "", response.strip())

        # 解析生成的 JSON 格式的知识点
        try:
            knowledge_points = json.loads(json_content)
            results = []
            for item in knowledge_points:
                for _, inner in item.items():  # 遍历外层唯一的 key
                    results.append((inner["knowledge"], inner["difficulty"], inner['question']))
            return results
            # return [(kp["knowledge"], kp["difficulty"]) for kp in knowledge_points]
        except json.JSONDecodeError:
            print("JSON 解析错误，请检查生成的内容格式--提取知识点并评估难度")
            return []

    def generate_questions_for_point(self, knowledge_point, original_question, difficulty, full_text):
        """根据知识点和难度生成适合的多种题型"""
        # 根据难度选择适合的题型
        if difficulty == "simple" or difficulty == "简单":
            question_types = ["multiple_choice"]
        elif difficulty == "medium" or difficulty == "中等":
            question_types = ["short_answer"]
        elif difficulty == "complex" or difficulty == "困难":
            question_types = ["open_discussion"]
        else:
            raise ValueError("未知难度级别")

        questions = []

        for q_type in question_types:
            # 从 prompts 中获取指定题型的 prompt 模板
            prompt_template = self.prompts[f"{q_type}"]

            prompt = prompt_template.format(
                full_text=full_text,
                knowledge_point=knowledge_point,
                original_question=original_question,
            )

            # 调用 run_chatgpt 函数生成问题和答案
            response = run_agent(prompt, model=self.model, num_gen=1, temperature=0.7)

            # 默认存 response
            record = {
                "knowledge": knowledge_point,
                "difficulty": difficulty,
                "question_type": q_type,
            }
            response = re.sub(r"```(?:json)?|```", "", response.strip())
            parsed = json.loads(response)
            # 🔸 如果是 multiple_choice，则尝试解析 JSON，并提取为 CSV 格式字段
            if q_type == "multiple_choice":
                # 安全获取选项，补齐到 4 个
                options = parsed.get("options", [])
                options = (options + ["", "", "", ""])[:4]

                # 清洗字段，处理逗号转义
                question_text = parsed.get("question", "").replace(",", "，")
                answer_letter = parsed.get("answer", "").strip().upper()

                # 构造 CSV 行字符串（字段顺序：question,A,B,C,D,answer）
                csv_record = f'"{question_text}",{options[0]},{options[1]},{options[2]},{options[3]},{answer_letter}'

                record["response"] = csv_record  # ✅ 添加 CSV 格式字段（便于后续保存）

            elif q_type in ["short_answer", "open_discussion"]:
                # 简答题和讨论题统一结构（字段相同）
                question_text = parsed.get("question", "").replace(",", "，")
                answer_text = parsed.get("answer", "").strip()

                record["response"] = {
                    "question": question_text,
                    "answer": answer_text
                }

            questions.append(record)

        return questions    


class ExpertAgent(BaseAgent):
    """专家 Agent，用于评估和改进试题质量"""

    def __init__(self, model="qwen"):
        super().__init__(name="ExpertAgent", model=model)
        from prompts.expert_prompts import EXPERT_PROMPTS_CN

        self.prompts = EXPERT_PROMPTS_CN

    def evaluate_and_refine_question(self, text, question_data, data_class):
        """
        评估并改进问题
        - question_data: 包含知识点、问题类型、初始问题和答案等信息的字典
        - data_class: 数据类别，用于决定扩展的深度和方式
        """
        knowledge_point = question_data["knowledge"]
        question_type = question_data["question_type"]

        # ✅ 构造标准输入文本：用于评估 or 改写
        if question_type == "multiple_choice":
            eval_input = question_data.get("response", "")

        elif question_type in ["short_answer", "open_discussion"]:
            response = question_data.get("response", {})
            question_text = response.get("question", "").strip()
            answer_text = response.get("answer", "").strip()
            eval_input = f"问题：{question_text}\n参考答案：{answer_text}"

        # ✅ Step 1: 试题质量评估
        expert_feedback = self.evaluate_quality(
            text, eval_input, knowledge_point, question_type, data_class
        )

        # ✅ Step 2: 根据需要进行改写
        if expert_feedback.get("requires_refinement", False):
            expert_feedback["refined_response"] = self.refine_response(
                text, eval_input, knowledge_point, data_class, expert_feedback
            )
        else:
            expert_feedback["refined_response"] = ""

        return expert_feedback  # 返回评估数据

    def evaluate_quality(self, text, response, 
                         knowledge_point, question_type, 
                         data_class="web"
                         ):
        """评估试题质量并判断是否需要改进"""
        prompt = self.prompts[f"evaluate_quality_{data_class}"].format(
            text=text,
            response=response,
            knowledge=knowledge_point,
            question_type=question_type,
        )

        evaluation_response = run_agent(prompt, model=self.model, num_gen=1, temperature=1)

        # 文本格式解析
        evaluation_response = re.sub(r"```(?:json)?|```", "", evaluation_response.strip())
        result = json.loads(evaluation_response)
        # 字段转小写 key（兼容模型大小写误差）
        result = {k.lower(): v for k, v in result.items()}
        # 6. 判断是否字段缺失或分数太低
        delete_data = any(
            result.get(k) is None or result.get(k) < 6
            for k in ["quality score", "relevance score", "consistency score"]
        )
        requires_refinement = result.get("quality score", 0) < 6

        # 返回结构
        return {
            "requires_refinement": requires_refinement,
            "delete_data": delete_data,
            "quality_score": result.get("quality score"),
            "relevance_score": result.get("relevance score"),
            "consistency_score": result.get("consistency score"),
            "improvement_suggestions": result.get("improvement suggestions", ""),
        }

    def refine_response(self, text, response, knowledge_point, data_class, expert_feedback):
        """
        根据评估反馈改进问题和答案。

        - text: 原始内容
        - response: 初始生成的问题和答案
        - knowledge_point: 知识点
        - data_class: 数据类别（如 books, articles, web）
        - expert_feedback: 评估反馈，包含改进建议
        """
        # 获取用于改进的 prompt
        prompt_template = self.prompts[f"refine_response_{data_class}"]

        # 填充 prompt 模板
        prompt = prompt_template.format(
            text=text,
            response=response,
            knowledge=knowledge_point,
            feedback=expert_feedback.get("feedback", " "),
        )

        # 调用大模型生成改进后的内容
        refined_response = run_agent(prompt, model=self.model, num_gen=1, temperature=1)
        refined_response = re.sub(r"```(?:json)?|```", "", refined_response.strip())
        parsed = json.loads(refined_response)
        return {
            "question": parsed.get("question", "").strip(),
            "answer": parsed.get("answer", "").strip()
        }


class VirtualTeacherAgent(BaseAgent):
    def __init__(self, name="VirtualTeacherAgent", model="qwen"):
        """初始化虚拟教师Agent"""
        super().__init__(name=name, model=model)
        from prompts.traininginstitute_prompts import CONVERSATION_PROMPTS_CN

        self.prompts = CONVERSATION_PROMPTS_CN

    def generate_thinking_chain(self, text, response, data_class):
        """生成思维链，用于引导学生思考和推理答案"""
        if isinstance(response, list):
            response = response[0] + ", " + response[1]
        # 获取思维链生成模板
        prompt_template = self.prompts.get(
            f"generate_chain_of_thought_{data_class}", {}
        )
        prompt = prompt_template.format(response=response)

        # 使用模型生成思维链
        thinking_chain = run_agent(prompt, model=self.model, num_gen=1, temperature=0.8)
        thinking_chain = re.sub(r"```(?:json)?|```", "", thinking_chain.strip())
        parsed = json.loads(thinking_chain)
        # 获取思维链
        formatted_thinking_chain = parsed.get("CoT", "").strip()
        # 返回结果
        return formatted_thinking_chain

    def convert_to_conversational_form(self, text, response, data_class):
        """将选择题转换为更自然的口语化对话形式"""
        # 获取对话转换模板
        prompt_template = self.prompts.get(f"convert_to_conversation_{data_class}", {})
        prompt = prompt_template.format(text=text, response=response)

        # 使用模型生成对话形式
        conversational_response = run_agent(
            prompt, model=self.model, num_gen=1, temperature=0.7
        )
        conversational_response = re.sub(r"```(?:json)?|```", "", conversational_response.strip())
        # 拼接问题和答案
        try:
            parsed_response = json.loads(conversational_response)
            question = parsed_response.get("input", "").strip()
            answer = parsed_response.get("output", "").strip()
            
            # 拼接问题和答案
            conversational_form = f"问题: {question}\n回答: {answer}"

        except json.JSONDecodeError as e:
            # 如果返回的结果不能被解析为 JSON，直接返回原始内容
            conversational_form = conversational_response

        # 返回拼接后的对话内容
        return conversational_form


    def cot_deepseek(self, response):
        """生成思维链，用于引导学生思考和推理答案"""
        # 获取思维链生成模板
        # prompt_template = self.prompts.get(
        #     f"generate_chain_of_thought_book", {}
        # )
        # prompt = prompt_template.format(response=response)

        prompt = response

        # 使用模型生成思维链
        thinking_chain = run_agent(prompt, model=self.model, num_gen=1, temperature=1)

        # 返回结果
        return thinking_chain

