import os
import sys
import json
import torch
import argparse
import logging
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.global_methods import run_agent  # 替换为你的推理函数
from data.dataset import get_dataloader

# === 日志配置 ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def extract_answer_from_tags(output_text):
    """
    提取 <think> 和 <answer> 内容。
    - 有 <think> 提取思考过程。
    - 有 <answer> 提取答案。
    - 没有 <answer>，但有 <think>，提取 think 后的剩余文本作为答案。
    - 两者都没有时，把整个文本作为答案。
    """
    think_match = re.search(r"<think>(.*?)</think>", output_text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL)

    thought_process = think_match.group(1).strip() if think_match else ""

    if answer_match:
        answer_candidate = answer_match.group(1).strip()
    elif think_match:
        after_think = output_text[think_match.end():].strip()
        answer_candidate = after_think
    else:
        answer_candidate = output_text.strip()

    return thought_process, answer_candidate


def extract_first_option(text):
    """优先提取 \boxed{}、<answer>、"answer": "X"、再 fallback 纯字母"""
    text_upper = text.strip()
<<<<<<< HEAD
    for pattern in [ 
        r"\\boxed\{([ABCD])\}",                # 匹配 \boxed{A}
        r"<answer>\s*([ABCD])\s*</answer>",    # 匹配 <answer>A</answer>
        r'"answer"\s*:\s*"([ABCD])"',          # 匹配 "answer": "A"
        r'"答案"\s*:\s*"([ABCD])"',            # ✅ 新增，匹配 "答案": "A" 这种中文的
        r"\b([ABCD])[\s\).，、。]?"             # fallback，直接找单字母
=======
    for pattern in [
        r"\\boxed\{([ABCD])\}",           # 匹配 \boxed{A}
        r"<answer>\s*([ABCD])\s*</answer>", # 匹配 <answer>A</answer>
        r'"answer"\s*:\s*"([ABCD])"',       # 匹配 "answer": "A"
        r"\b([ABCD])[\s\).，、。]?"          # fallback，直接找单字母
>>>>>>> 9ce0f14f46524713196e7f72c74d8dd781d1007f
    ]:
        match = re.search(pattern, text_upper)
        if match:
            return match.group(1)
    return None


def is_valid_option(answer):
    return answer.strip() in ["A", "B", "C", "D"] or extract_first_option(answer) is not None

def infer_answer_with_gpt4(model_output):
    """如果答案不是有效选项，调用 GPT-4 进行推理"""
    logger.info(f"调用 GPT-4 进行答案修正: {model_output}")
    prompt = f"""
    你是一个严格的评分员，请根据以下模型输出，判断模型最有可能选择的选项（A/B/C/D）。
    - **模型原始输出**: {model_output}
    
    题目为单选题，请直接输出最可能的选项（A/B/C/D），不需要解释。
    """
    try:
        gpt4_answer = run_agent(prompt, model="qwen")
        if is_valid_option(gpt4_answer):
            return gpt4_answer
    except Exception as e:
        logger.error(f"GPT-4 推理失败: {e}")
    
    return "未知"  # GPT-4 失败时返回 "未知"

def generate_text(model, tokenizer, input_texts, batch_msg, max_new_tokens=8192, temperature=0.6):
    """生成并解析 batch 结果"""
    # inputs1 = tokenizer(input_texts, padding=True, truncation=True, max_length=2048, return_tensors="pt").to(model.device)

    rendered_texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in batch_msg]
    inputs = tokenizer(rendered_texts, padding=True, return_tensors="pt").to(model.device)
    
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,   
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )
    outputs = tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)
    outputs = [o.strip() for o in outputs]
    return outputs

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True
    ).eval()
    return model, tokenizer

def load_existing_ids(save_file):
    if not os.path.exists(save_file): return set()
    with open(save_file, 'r', encoding='utf-8') as f:
        return set(int(json.loads(line)["id"]) for line in f if line.strip())

def evaluate_and_save(model, tokenizer, dataloader, total_batches, output_path, batch_size=8, temperature=0.3):
    """
    评估并保存结果到 JSONL，支持 batch 推理、断点恢复。
    """
    save_file = os.path.join(output_path, "results.jsonl")
    os.makedirs(output_path, exist_ok=True)

    done_ids = load_existing_ids(save_file)
    logger.info(f"🔹 已完成 {len(done_ids)} 条，将跳过这些样本...")

    with open(save_file, "a", encoding="utf-8") as fout, tqdm(desc="推理中", unit="sample") as pbar:
        for batch in dataloader:
            prompts = [item["prompt"] for item in batch]
            messages = [item["messages"] for item in batch]
            answers = [item["answer"] for item in batch]
            ids = [item["id"] for item in batch]  # ⭐ 取真实 id

            batch_prompts = []
            batch_msg = []
            batch_infos = []

            for prompt, msg, ref, id_ in zip(prompts, messages, answers, ids):
                if id_ in done_ids:
                    continue
                batch_prompts.append(prompt)
                batch_msg.append(msg)
                batch_infos.append((id_, prompt, msg, ref))

            if not batch_prompts:
                continue  # 这个 batch 全跳过了

            outputs = generate_text(model, tokenizer, batch_prompts, batch_msg, temperature=temperature)

            for (id_, prompt, msg, ref), output in zip(batch_infos, outputs):
                thought, candidate = extract_answer_from_tags(output)
                candidate = candidate.strip()

                if candidate in ["A", "B", "C", "D"]:
                    pred = candidate
                else:
                    extracted = extract_first_option(candidate)
                    if extracted:
                        pred = extracted
                    else:
                        pred = infer_answer_with_gpt4(candidate)

                record = {
                    "id": id_,
                    "prompt": prompt,
                    "message": msg,
                    "reference": ref,
                    "raw_output": output,
                    "predicted": pred,
                    "correct": pred == ref,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                pbar.update(1)



def main():
    parser = argparse.ArgumentParser()
<<<<<<< HEAD
    parser.add_argument("--model_path", type=str, default="/mnt/sda/wyp/models/Qwen3-8B", help="本地模型路径")
    parser.add_argument("--input_file", type=str, default="/mnt/sda/wyp/forestllm-main/forest_eval/forest_book_mcq_1k3.csv", help="输入数据文件")
=======
    parser.add_argument("--model_path", type=str, default="/mnt/sda/wyp/models/DeepSeek-R1-Distill-Qwen-7B", help="本地模型路径")
    parser.add_argument("--input_file", type=str, default="/mnt/sda/wyp/forestllm-main/outputs/compare_subsets/books_mcq_1k5_v1.csv", help="输入数据文件")
>>>>>>> 9ce0f14f46524713196e7f72c74d8dd781d1007f
    parser.add_argument("--output_dir", type=str, default="/mnt/sda/wyp/forestllm-main/outputs/eval/", help="评估结果存储目录")
    parser.add_argument("--task_type", type=str, choices=["mcq", "qa"], default="mcq", help="任务类型")
    parser.add_argument("--evaluation_method", type=str, choices=["metrics", "gpt4", "manual"], default="metrics", help="评估方式")
    parser.add_argument("--batch_size", type=int, default=4, help="批量推理大小")
<<<<<<< HEAD
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.6, help="生成温度")
    parser.add_argument("--model_mode", type=str, default='normal', help="模板选择")
=======
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.6, help="生成温度")
    parser.add_argument("--model_mode", type=str, default='cot', help="模板选择")
>>>>>>> 9ce0f14f46524713196e7f72c74d8dd781d1007f

    args = parser.parse_args()
    # /mnt/sda/wyp/forestllm-main/forest_eval/forest_zero_shot.csv
    # /mnt/sda/wyp/forestllm-main/forest_eval/book/eval_multiple_choice_filtered.csv
    # /mnt/sda/wyp/models/checkpoint-12277
    # /mnt/sda/wyp/models/qwen25
    # /mnt/sda/wyp/models/llama
    # /mnt/sda/wyp/models/Qwen3-8B-Base
<<<<<<< HEAD
    # /mnt/sda/wyp/models/DeepSeek-R1-Distill-Qwen-7B
=======
>>>>>>> 9ce0f14f46524713196e7f72c74d8dd781d1007f
    model, tokenizer = load_model(args.model_path)
    tag = os.path.basename(args.model_path).split("/")[-1]
    eval_split = os.path.splitext(os.path.basename(args.input_file))[0]
    output_path = os.path.join(args.output_dir, tag, eval_split)

    dataloader, total_batches = get_dataloader(
        args.input_file,
        batch_size=args.batch_size,
        task_type=args.task_type,
        model_mode=args.model_mode
    )

    evaluate_and_save(model, tokenizer, dataloader, total_batches, output_path, temperature=args.temperature)

if __name__ == "__main__":
    main()
<<<<<<< HEAD


# CUDA_VISIBLE_DEVICES=3 python eval_new.py --model_path /mnt/sda/wyp/models/qwen3_8b_sft_ep3 --input_file /mnt/sda/wyp/forestllm-main/forest_eval/compare_subsets/forest_zero_shot_v1.csv --temperature 0.6
# CUDA_VISIBLE_DEVICES=2 python eval_new.py --model_path /mnt/sda/wyp/models/Qwen3-8B --input_file /mnt/sda/wyp/forestllm-main/forest_eval/compare_subsets/forest_zero_shot_v1.csv --temperature 0.6
=======
>>>>>>> 9ce0f14f46524713196e7f72c74d8dd781d1007f
