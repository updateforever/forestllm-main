import os
import json
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PROMPT_COT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"""

def messages_to_prompt(messages):
    """将 messages 转为 prompt 字符串（ChatML 格式）"""
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"<|system|>\n{content}\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"
    prompt += "<|assistant|>\n"
    return prompt

class CustomDataset(Dataset):
    """自定义数据集，支持 MCQ 和 QA 任务，返回 messages 与 prompt，支持 bucket 排序"""

    def __init__(self, file_path, task_type="mcq", num_buckets=10, model_mode='normal'):
        self.file_path = file_path
        self.task_type = task_type
        self.num_buckets = num_buckets
        self.data = self._load_data(model_mode)
        self._sort_by_length()
        self.buckets = self._create_buckets()

    def _load_data(self, model_mode='normal'):
        file_ext = os.path.splitext(self.file_path)[1].lower()
        data_list = []

        if file_ext == ".csv":
            with open(self.file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if self.task_type == "mcq":
                        question = row["question"]
                        options = f"A) {row['A']}\nB) {row['B']}\nC) {row['C']}\nD) {row['D']}"
                        answer = row["answer"].strip().upper()
                        id_ = int(row["id"]) if "id" in row else None  # 读取 id 字段
                        
                        system_prompt = "你是一个专业的林业智能问答助手。" if model_mode == 'normal' else PROMPT_COT
                        user_prompt = f"请阅读下列单选题，并在答案栏中只填写选择的字母，例如：\"answer\": \"C\"。\n 单选题：{question}\n{options}\n"
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                        prompt = messages_to_prompt(messages)

                        data_list.append({
                            "id": id_,
                            "messages": messages,
                            "prompt": prompt,
                            "answer": answer,
                            "length": sum(len(m["content"]) for m in messages)
                        })

                    elif self.task_type == "qa":
                        if "input" in row and "reference" in row:
                            input_text = row["input"]
                            answer = row["reference"]
                            messages = [
                                {"role": "system", "content": "你是一个专业的问答助手"},
                                {"role": "user", "content": input_text}
                            ]
                            prompt = messages_to_prompt(messages)
                            data_list.append({
                                "messages": messages,
                                "prompt": prompt,
                                "answer": answer,
                                "length": len(input_text)
                            })
                        else:
                            raise ValueError("CSV 结构错误: 需要 `input` 和 `reference` 字段")
        else:
            raise ValueError("不支持的文件格式，请使用 CSV")

        return data_list

    def _sort_by_length(self):
        """按文本长度排序"""
        self.data.sort(key=lambda x: x["length"])

    def _create_buckets(self):
        """将数据划分为多个长度相近的 bucket"""
        bucket_size = max(1, len(self.data) // self.num_buckets)
        return [self.data[i:i + bucket_size] for i in range(0, len(self.data), bucket_size)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_batches(self, batch_size):
        """生成 batch，每个 batch 里数据长度尽量接近"""
        batches = []
        for bucket in self.buckets:
            for i in range(0, len(bucket), batch_size):
                batch = bucket[i:i + batch_size]
                if batch:
                    batches.append(batch)
        return batches

class IterableDatasetWrapper(torch.utils.data.IterableDataset):
    """封装 batch 生成器，使其兼容 PyTorch DataLoader"""
    def __init__(self, batches):
        super().__init__()
        self.batches = batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

def get_dataloader(file_path, batch_size=4, task_type="mcq", model_mode='normal'):
    dataset = CustomDataset(file_path, task_type=task_type, model_mode=model_mode)
    batches = dataset.get_batches(batch_size)
    return DataLoader(IterableDatasetWrapper(batches), batch_size=None, shuffle=False), len(batches)
