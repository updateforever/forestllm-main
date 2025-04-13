import os
import json
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    """ 自定义数据集，支持 MCQ 和 QA 任务，并采用 bucket sampling 进行 batch 划分 """

    def __init__(self, file_path, task_type="mcq", num_buckets=10, model_mode='normal'):
        self.file_path = file_path
        self.task_type = task_type
        self.num_buckets = num_buckets  # 多少个长度桶
        self.inputs, self.references, self.lengths = self._load_data(model_mode)
        
        # **按长度排序**
        self._sort_by_length()
        
        # **按照 bucket 采样**
        self.buckets = self._create_buckets()

    def _load_data(self, model_mode='normal'):
        """从 JSON 或 CSV 读取数据"""
        file_ext = os.path.splitext(self.file_path)[1].lower()
        inputs, references, input_lengths = [], [], []

        if file_ext == ".json":
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            inputs, references = data["inputs"], data["references"]
            input_lengths = [len(text) for text in inputs]

        elif file_ext == ".csv":
            with open(self.file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if self.task_type == "mcq":
                        question = row["question"]
                        options = f"A) {row['A']}\nB) {row['B']}\nC) {row['C']}\nD) {row['D']}"
                        answer = row["answer"].strip().upper()

                        if model_mode=='normal':
                            prompt = f"请阅读以下问题，并直接给出正确答案的选项。若需思考，请尽量进行短思考，并快速给出最终答案。\n\n问题：{question}\n{options}\n答案："
                        else:
                            # prompt = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. \
                            #     The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. \
                            #         The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>\
                            #         请解决以下问题。请将思考过程写在 <think> 和 </think> 标签中，并将最终答案写在 <answer> 和 </answer> 标签中。\n\n问题：{question}\n{options}\n答案："
                            
                            prompt = f"""Please reason step by step, and put your final answer with \boxed{{}}.(Don't make your reasoning too long)\nUser: 请阅读以下问题，并直接给出正确答案的选项。若需思考，请尽量进行短思考，并快速给出最终答案。\n问题：{question}\n{options}\nAssistant: <think>\n"""
                            
                        inputs.append(prompt)
                        references.append(answer)
                        input_lengths.append(len(prompt))

                    elif self.task_type == "qa":
                        if "input" in row and "reference" in row:
                            inputs.append(row["input"])
                            references.append(row["reference"])
                            input_lengths.append(len(row["input"]))
                        else:
                            raise ValueError("CSV 结构错误: 需要 `input` 和 `reference` 字段")

        else:
            raise ValueError("不支持的文件格式，请使用 JSON 或 CSV 格式。")

        return inputs, references, input_lengths

    def _sort_by_length(self):
        """按文本长度排序"""
        sorted_indices = np.argsort(self.lengths)
        self.inputs = [self.inputs[i] for i in sorted_indices]
        self.references = [self.references[i] for i in sorted_indices]
        self.lengths = [self.lengths[i] for i in sorted_indices]

    def _create_buckets(self):
        """将数据划分为多个长度相近的 bucket"""
        bucket_size = len(self.inputs) // self.num_buckets  # 计算每个 bucket 大小
        buckets = [
            list(zip(self.inputs[i:i + bucket_size], self.references[i:i + bucket_size]))
            for i in range(0, len(self.inputs), bucket_size)
        ]
        # np.random.shuffle(buckets)  # 只打乱 bucket 的顺序，不打乱 bucket 内部的长度排序
        return buckets

    def __len__(self):
        """返回数据总数"""
        return len(self.inputs)

    def get_batches(self, batch_size):
        """生成 batch，每个 batch 里数据长度尽量接近"""
        batches = []
        for bucket in self.buckets:
            for i in range(0, len(bucket), batch_size):
                batch = bucket[i:i + batch_size]
                if len(batch) > 0:
                    batch_inputs, batch_references = zip(*batch)
                    batches.append((list(batch_inputs), list(batch_references)))
        
        # np.random.shuffle(batches)  # 再次打乱 batch 的顺序
        return batches

class IterableDatasetWrapper(torch.utils.data.IterableDataset):
    """封装 batch 生成器，使其兼容 PyTorch DataLoader"""
    def __init__(self, batches):
        super().__init__()
        self.batches = batches

    def __iter__(self):
        """DataLoader 直接迭代 batch"""
        for batch in self.batches:
            yield batch  # batch 本身已经是一个列表，无需再拆

def get_dataloader(file_path, batch_size=4, task_type="mcq", model_mode='normal'):
    """使用 `CustomDataset` 进行数据加载，确保 batch_size 统一"""
    dataset = CustomDataset(file_path, task_type=task_type, model_mode=model_mode)
    batches = dataset.get_batches(batch_size)  # 这里 batch_size 已经生效
    total_batches = len(batches)  # ✅ 计算 batch 总数

    return DataLoader(
        IterableDatasetWrapper(batches),  # ✅ 让 DataLoader 直接迭代 batch
        batch_size=None,  # ✅ 让 DataLoader 不再分 batch
        shuffle=False  # ✅ batch 内部已经打乱过，这里不再打乱
    ), total_batches 


'''
def load_data(file_path, batch_size=4, task_type="mcq"):
    """按批次加载数据，确保同批数据长度接近，以减少 padding 影响。

    - `mcq`: 适用于多选题（CSV 格式：eval_id, question, A, B, C, D, answer）。
    - `qa`: 适用于问答任务（JSON 或 CSV 格式）。
    - **优化点**: 按文本长度排序，使每批数据长度接近，减少 `padding` 造成的计算损耗。
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    inputs, references, input_lengths = [], [], []

    if file_ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        inputs, references = data["inputs"], data["references"]
        input_lengths = [len(text) for text in inputs]

    elif file_ext == ".csv":
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if task_type == "mcq":
                    question = row["question"]
                    options = f"A) {row['A']}\nB) {row['B']}\nC) {row['C']}\nD) {row['D']}"
                    answer = row["answer"].strip().upper()
                    
                    prompt = f"请阅读以下问题，并直接给出正确答案的选项。\n\n示例：\n问题：2+2 等于多少？\nA) 1\nB) 2\nC) 3\nD) 4\n答案：D\n\n问题：{question}\n{options}\n答案："
                    
                    inputs.append(prompt)
                    references.append(answer)
                    input_lengths.append(len(prompt))

                elif task_type == "qa":
                    if "input" in row and "reference" in row:
                        inputs.append(row["input"])
                        references.append(row["reference"])
                        input_lengths.append(len(row["input"]))
                    else:
                        raise ValueError("CSV 结构错误: 需要 `input` 和 `reference` 字段")

    else:
        raise ValueError("不支持的文件格式，请使用 JSON 或 CSV 格式。")

    # **1️⃣ 按输入文本长度排序（从短到长）**
    sorted_indices = np.argsort(input_lengths)
    inputs = [inputs[i] for i in sorted_indices]
    references = [references[i] for i in sorted_indices]
    input_lengths = [input_lengths[i] for i in sorted_indices]

    # **2️⃣ 按 `batch_size` 进行分组，确保同 batch 的长度接近**
    batches = [
        (inputs[i:i + batch_size], references[i:i + batch_size])
        for i in range(0, len(inputs), batch_size)
    ]

    # **3️⃣ 重新打乱 batch 顺序，防止模型学到长度模式**
    np.random.shuffle(batches)

    # **4️⃣ 逐批返回数据**
    for batch in batches:
        yield batch
'''