# from global_methods import run_chatgpt
import torch

# from agent import BaseAgent
from agents.agent import BaseAgent
from utils.toolkit import *


class SimulatedLearner(BaseAgent):
    """模拟考生 Agent，用于回答问题"""

    def __init__(self, model_api="qwen", model_paths=None, model_platforms=None):
        """
        初始化模拟考生Agent
        - model_apis: 模型API
        - model_paths: 路径
        - model_platforms: 平台
        """
        super().__init__(name="SimulatedLearner", model=None)
        self.models = []
        self.tokenizers = []
        self.model_names = []
        from prompts.student_prompts import SIMULATE_ANSWER_CN

        self.prompt = SIMULATE_ANSWER_CN

        # 加载多个模型
        if model_paths and model_platforms:
            for model_path, model_platform in zip(model_paths, model_platforms):
                self.load_model(model_platform, model_path)
        else:
            # 默认处理模型API
            self.models.append(model_api)
            self.tokenizers.append(None)

    def load_model(self, platform, model_path):
        """根据平台加载模型"""
        if platform == "huggingface":
            # 加载 HuggingFace 模型
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizers.append(
                AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            )
            self.models.append(
                AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda",
                    trust_remote_code=True,
                )
            )
            self.model_names.append(model_path)
        elif platform == "modelscope":
            # 加载 ModelScope 模型
            from modelscope import AutoModelForCausalLM, AutoTokenizer

            # if 'qwenvl' in model_path:
            #      from modelscope import Qwen2VLForConditionalGeneration
            #      Qwen2VLForConditionalGeneration.from_pretrained(
            #         model_path, torch_dtype="auto", device_map="auto"
            #     )
            self.tokenizers.append(
                AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            )
            self.models.append(
                AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype="auto",
                    device_map="auto",
                    trust_remote_code=True,
                )
            )
            self.model_names.append(model_path)

    def answer_question(self, response_data):
        """
        生成模拟考生的回答
        - question_data: 包含知识点、问题类型、问题和答案等信息的字典
        """
        # 从 response_data 提取问题部分
        question, _ = extract_question(response_data)

        # 准备生成回答的 prompt
        prompt_template = self.prompt
        prompt = prompt_template.format(question=question)

        # 生成模拟考生的回答
        learner_answer = self.run_inference(prompt)

        # 返回回答
        return learner_answer

    def answer_questions_batch(self, response_data_list):
        """批量生成模拟考生的回答"""
        questions = [extract_question(data['response'])[0] for data in response_data_list]
        prompts = [self.prompt.format(question=q) for q in questions]

        all_answers = []
        for model, tokenizer in zip(self.models, self.tokenizers):
            if tokenizer:
                if 'mini' in model.model_dir or 'llama' in model.model_dir:
                    tokenizer.pad_token = tokenizer.eos_token
                inputs = tokenizer(prompts, padding=True, truncation=True, max_length=256, padding_side='left', return_tensors="pt").to(model.device)
                input_length = inputs["input_ids"].shape[1]
                outputs = model.generate(**inputs, max_length=512, no_repeat_ngram_size=2)
                answers = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
                all_answers.append(answers)
            else:
                all_answers.append([run_agent(prompt, model=model) for prompt in prompts])

        # 转置以便每个问题有一个回答
        final_answers = list(map(list, zip(*all_answers)))
        return final_answers


    def run_inference(self, prompt):
        """运行推理，生成回答"""
        answers = []

        for model, tokenizer in zip(self.models, self.tokenizers):
            if tokenizer:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs, max_length=500, no_repeat_ngram_size=2
                )
                answer = tokenizer.decode(
                    outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
                )
                answers.append(answer)
            else:
                # 如果是自定义API模型，直接调用 run_agent
                answers.append(run_agent(prompt, model=model))

        return answers  # 返回多个模型的输出


# 主程序调用示例
# if __name__ == "__main__":
#     learner_agent = SimulatedLearner(
#         prompt_file="/home/wyp/project/ForestLLM/prompts/student_prompts.json",
#         model_api=["qwen"],
#         model_paths=["/home/wyp/project/swift/models/qwen25_7b_ins"],
#         model_platforms=['modelscope'],
#     )

#     # 示例问题和答案
#     question_data = {
#         "response": "植物光合作用的主要原料是什么？",
#         "expert_feedback": {"refined_response": "二氧化碳是植物光合作用的主要原料。"}
#     }

#     learner_answer = learner_agent.answer_question(question_data)

#     # 输出多个模型的答案
#     for answer in learner_answer:
#         print(f"回答: {answer}")
