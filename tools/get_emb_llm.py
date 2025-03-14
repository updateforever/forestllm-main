import json
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import argparse
from transformers import AutoTokenizer, AutoModel
from accelerate import infer_auto_device_map

# 生成唯一 ID
def generate_entry_id(entry):
    """通过 JSON 序列化方式生成唯一 ID"""
    entry_str = json.dumps(entry, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(entry_str.encode("utf-8")).hexdigest()

def load_model(model_path):
    print(f"🔄 Loading model from {model_path} ...")

    # **自动检测模型类型**
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # **避免 padding 错误（LLaMA 没有 pad_token）**
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # **强制右截断**
    tokenizer.truncation_side = "right"

    # **尝试用 `device_map="auto"` 自动分配**
    try:
        model = AutoModel.from_pretrained(model_path, device_map="auto")
        device = next(model.parameters()).device
        print("Model loaded with `device_map='auto'` (multi-GPU enabled)")
    except TypeError:
        # **如果 `auto` 失败（一般是 BERT），则手动 `to(device)`**
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModel.from_pretrained(model_path).to(device)
        print("Model loaded and moved to device manually (BERT / smaller models)")

    return tokenizer, model, device

# 读取数据（支持原始数据 & 生成数据）
def load_data(mode, original_file, generated_file, generated_ids=None):
    """ 根据模式 ('original' or 'generated') 读取数据 """
    texts, ids, knowledge_texts = [], [], []

    if mode == "original":
        print(f"📂 Loading original dataset from {original_file} ...")
        with open(original_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Processing Original JSONL"):
                try:
                    data = json.loads(line)
                    if "text" in data:
                        text = data["text"]
                        unique_id = generate_entry_id(data)
                        # # **✅ 新增：ID 过滤**
                        # if generated_ids and unique_id not in generated_ids:
                        #     continue  # 跳过无关数据
                        texts.append(text)
                        ids.append(unique_id)
                except json.JSONDecodeError:
                    print("⚠️ Skipping invalid JSON line.")
    
    elif mode == "generated":
        print(f"📂 Loading generated dataset from {generated_file} ...")
        with open(generated_file, "r", encoding="utf-8") as f:
            generated_data = json.load(f)
        for item in tqdm(generated_data, desc="Processing Generated JSON"):
            texts.append(item['question_setter']["response"])
            ids.append(item["id"])
            knowledge_texts.append(item["question_setter"].get("knowledge", ""))  # 可能没有 knowledge

    print(f"✅ Total {len(texts)} documents loaded for '{mode}' dataset.")
    return texts, ids, knowledge_texts

# 计算文本嵌入
def get_embedding(texts, tokenizer, model, device, batch_size=2, max_length=2048):
    """ 使用 LLaMA 计算文本嵌入 """
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="🔍 Generating embeddings"):
        batch_texts = texts[i:i + batch_size]
        
        # 对文本进行编码
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
        encoded_input = {key: value.to(device) for key, value in encoded_input.items()}  # 传输到 GPU
        
        # 获取模型输出
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # 取最后一层 [CLS] token 向量作为嵌入
        sentence_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(sentence_embeddings)

    return np.vstack(embeddings)

# 保存嵌入到文件
def save_embeddings(embeddings, ids, mode, output_dir, prefix="response"):
    """ 存储 NumPy & CSV 格式的嵌入数据 """
    npy_path = f"{output_dir}/llama_embeddings_{mode}_{prefix}.npy"
    csv_path = f"{output_dir}/llama_embeddings_{mode}_{prefix}.csv"

    np.save(npy_path, embeddings)
    print(f"💾 Embeddings saved to {npy_path}")

    df = pd.DataFrame(embeddings)
    df.insert(0, "id", ids)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"📜 CSV file saved to {csv_path}")

# 主函数
def main():
    """ 主入口函数，支持动态路径 """
    parser = argparse.ArgumentParser(description="Extract embeddings using LLaMA & BERT")
    
    # 可配置的路径
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the LLaMA/BERT model directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output files")
    parser.add_argument("--original_file", type=str, required=True,
                        help="Path to the original dataset JSONL file")
    parser.add_argument("--generated_file", type=str, required=True,
                        help="Path to the generated dataset JSON file")
    parser.add_argument("--max_length", type=int, default=None, 
                        help="Max sequence length (auto-detected)")
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="batch size")
    # 运行模式
    parser.add_argument("--mode", type=str, choices=["original", "generated", "both"], default="both",
                        help="Choose which dataset to process: 'original', 'generated', or 'both'")

    args = parser.parse_args()

    # 确保输出目录存在
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    tokenizer, model, device = load_model(args.model_path)

    # **自动调整 max_length**
    if args.max_length is None:
        model_type = model.config.model_type
        if "bert" in model_type or "roberta" in model_type:
            args.max_length = 512  # BERT & RoBERTa 模型默认 512
        elif "llama" in model_type or "gpt" in model_type:
            args.max_length = 2048  # LLaMA / GPT 默认 2048
        else:
            args.max_length = 1024  # 其他模型默认 1024
    print(f"⚙️ Using max_length={args.max_length} for model type: {model_type}")

    # 处理原始数据
    if args.mode in ["original", "both"]:
        # 读取生成数据的 ID
        print("🔍 Extracting IDs from generated data...")
        with open(args.generated_file, "r", encoding="utf-8") as f:
            generated_data = json.load(f)
        generated_ids = {item["id"] for item in generated_data}
        print(f"✅ Found {len(generated_ids)} IDs in generated data.")

        # 处理 `text` 字段
        texts, ids, _ = load_data("original", args.original_file, args.generated_file, generated_ids)
        embeddings = get_embedding(texts, tokenizer, model, device, args.batch_size, args.max_length)
        save_embeddings(embeddings, ids, "original", args.output_dir, "text")

    # 处理生成数据
    if args.mode in ["generated", "both"]:
        texts, ids, knowledge_texts = load_data("generated", args.original_file, args.generated_file)

        # 处理 `response` 字段
        embeddings = get_embedding(texts, tokenizer, model, device, args.batch_size, args.max_length)
        save_embeddings(embeddings, ids, "generated", args.output_dir, "response")

        # 处理 `knowledge` 字段
        knowledge_embeddings = get_embedding(knowledge_texts, tokenizer, model, device, args.batch_size, args.max_length)
        save_embeddings(knowledge_embeddings, ids, "generated", args.output_dir, "knowledge")

    print("✅ All processing complete!")


# 运行入口
if __name__ == "__main__":
    main()


'''
python get_emb_llm.py \
    --model_path "/home/wyp/project/swift/models/llama_3_1_8b_ins/" \
    --output_dir "outputs/emb_data/" \
    --original_file "mateinfo/merged_org_data.jsonl" \
    --generated_file "sft_output_all_250220.json" \
    --mode both

python tools/get_emb_llm.py \
    --model_path "/home/wyp/project/forest/forestllm-main/models/bert-base-chinese/" \
    --output_dir "outputs/emb_data/bert/" \
    --original_file "mateinfo/merged_org_data.jsonl" \
    --generated_file "outputs/emb_data/sft_output_all_250220.json" \
    --batch_size 32 \
    --mode both    
'''

