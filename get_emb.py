import json
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

# 设置输入文件路径
input_file = "/home/wyp/project/ForestLLM/data/mateinfo/books-1113.jsonl"  # 你的真实数据文件
output_file = "/home/wyp/project/forest/forestllm-main/outputs/emb_data/embeddings_hashed.npy"  # 嵌入特征的 NumPy 输出文件
csv_output = "/home/wyp/project/forest/forestllm-main/outputs/emb_data/embeddings_hashed.csv"  # 以 CSV 格式存储（可选）
generated_data_file = "qwen_book_output.json"  # 生成数据的 JSON 文件

# 加载 BERT 模型和 Tokenizer
# Load model directly
model_name = "/home/wyp/project/forest/forestllm-main/models/bert-base-chinese"  # 使用 BERT 中文模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 设置设备（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 读取生成数据的 ID（从 qwen_book_output.json 文件）
print("Reading generated data IDs...")
with open(generated_data_file, "r", encoding="utf-8") as f:
    generated_data = json.load(f)

# 提取所有生成数据的 ID，存储在集合中
generated_ids = {item['id'] for item in generated_data}
print(f"Total generated data IDs: {len(generated_ids)}")

# 生成唯一 ID
def generate_entry_id(entry):
    """通过 JSON 序列化方式生成唯一 ID"""
    entry_str = json.dumps(entry, sort_keys=True, ensure_ascii=False)  # 排序并确保不丢失中文字符
    return hashlib.sha256(entry_str.encode("utf-8")).hexdigest()

# 读取原始数据（真实数据）
texts = []
hashed_ids = []
print("Reading original data (real_data.jsonl)...")
with open(input_file, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Processing JSONL"):
        try:
            data = json.loads(line)  # 解析 JSONL 行
            if "text" in data:
                text = data["text"]
                # 生成原始文本的唯一 ID（通过生成 entry_id 的方式）
                unique_id = generate_entry_id(data)
                # 如果该 ID 不在生成数据的 ID 集合中，跳过
                if unique_id not in generated_ids:
                    continue
                texts.append(text)  # 添加有效文本
                hashed_ids.append(unique_id)  # 添加哈希 ID
        except json.JSONDecodeError:
            print("Skipping invalid JSON line.")

print(f"Total documents loaded: {len(texts)}")

# 计算文本嵌入
def get_embedding(texts, batch_size=16):
    """ 使用 Transformer 计算文本嵌入 """
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i + batch_size]
        # 对文本进行编码
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=2048)
        encoded_input = {key: value.to(device) for key, value in encoded_input.items()}  # 移动到GPU

        # 获取模型输出
        with torch.no_grad():  # 禁用梯度计算
            model_output = model(**encoded_input)

        # 获取 [CLS] token 的向量表示作为句子级别的嵌入
        sentence_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] 是第 0 个 token
        embeddings.append(sentence_embeddings)

    return np.vstack(embeddings)

# 计算文本的嵌入
print("Computing embeddings...")
embeddings = get_embedding(texts)

# 保存嵌入为 NumPy 文件
np.save(output_file, embeddings)
print(f"Embeddings saved to {output_file}")

# 可选：保存为 CSV 以便查看
df = pd.DataFrame(embeddings)
df.insert(0, "text", texts)  # 在 CSV 中保留原始文本
df.insert(1, "hash_id", hashed_ids)  # 添加哈希 ID 列
df.to_csv(csv_output, index=False, encoding="utf-8")
print(f"CSV file saved to {csv_output}")

print("✅ Processing complete!")
