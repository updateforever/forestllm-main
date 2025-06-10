import os
import sys
import json
import torch
import argparse
import logging
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
from utils.global_methods import *  # 确保 GPT-4 评估可用
from tqdm import tqdm 
from data.dataset import get_dataloader
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_device():
    """自动选择最佳计算设备"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # Mac 上的 Metal 设备
    else:
        return "cpu"

def load_model(model_path, temperature=1.0):
    """使用 Hugging Face Transformers 加载本地模型"""
    logger.info(f"正在加载本地模型: {model_path}")
    device = get_device()

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    # print(tokenizer.truncation_side)  # 输出 "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 解决 padding 问题

    # 加载模型（自动检测 `.safetensors`）
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto",  # 自动分配到 GPU 或 CPU
        torch_dtype="auto",
        trust_remote_code=True  # 仅使用本地代码
    ).eval()  # 设为 eval 模式，避免训练时的 Dropout

    logger.info(f"模型成功加载至 {device}， temperature={temperature}")
    return model, tokenizer


def extract_answer_from_tags(output_text):
    """
    从生成结果中提取 `<think>` 和 `<answer>` 内容。
    支持部分缺失情况：
    - 有完整的 <think>...</think> 和 <answer>...</answer>：正常解析
    - 缺失 <answer>：回退到 think 后的文本尝试提取
    - 缺失 <think>：仅提取 answer 标签
    - 全部缺失：原样返回
    """
    output_text = output_text.strip()
    think_match = re.search(r"<think>(.*?)</think>", output_text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL)

    thought_process = None
    answer_candidate = None

    if think_match:
        thought_process = think_match.group(1).strip()
    if answer_match:
        answer_candidate = answer_match.group(1).strip()

    # 如果只有 <think>，则尝试获取它后面的内容作为 answer_candidate
    if think_match and not answer_match:
        after_think = output_text[think_match.end():].strip()
        answer_candidate = after_think

    # 如果两个都没有，就返回原始内容
    if not thought_process and not answer_candidate:
        return None, output_text

    return thought_process, answer_candidate or ""

def extract_first_option(text):
    """
    从文本中提取第一个有效选项（A/B/C/D），
    支持格式：A、A)、A.、选项A、答案是A 等。
    """
    short_text = text.strip()[:100].upper()
    match = re.search(r"\b([ABCD])[\s\).，、。]?", short_text)
    if match:
        return match.group(1)
    return None

def extract_first_option(text):
    """
    从文本中提取第一个有效选项（A/B/C/D），按以下优先级：
    1. 提取 \boxed{A} 形式中的内容；
    2. 提取 <answer>A</answer> 形式中的内容；
    3. 从文本开头前100字符中解析 A/B/C/D。
    """
    text_upper = text.strip().upper()

    # 1️⃣ 尝试提取 \boxed{A}
    boxed_match = re.search(r"\\boxed\{([ABCD])\}", text)
    if boxed_match:
        return boxed_match.group(1)

    # 2️⃣ 尝试提取 <answer>A</answer>
    tag_match = re.search(r"<answer>\s*([ABCD])\s*</answer>", text)
    if tag_match:
        return tag_match.group(1)

    # 3️⃣ 回退到前100字符中搜索 A/B/C/D
    short_text = text[:100]
    fallback_match = re.search(r"\b([ABCD])[\s\).，、。]?", short_text)
    if fallback_match:
        return fallback_match.group(1)

    # 都失败返回 None
    return None

def is_valid_option(answer):
    """
    判断是否为有效的选项 A/B/C/D。
    - 优先尝试精确匹配
    - 再使用宽松规则提取
    """
    cleaned = answer.strip().upper()
    if cleaned in ["A", "B", "C", "D"]:
        return True
    return extract_first_option(cleaned) is not None

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

def extract_answer(output_text):
    """提取答案 (从 <think> 标签中解析)"""
    start_tag, end_tag = "<think>", "</think>"
    start_idx, end_idx = output_text.find(start_tag), output_text.find(end_tag)
    if start_idx != -1 and end_idx != -1:
        return output_text[start_idx + len(start_tag):end_idx].strip()
    return output_text.strip()

def compute_mcq_accuracy(predictions, references):
    """计算多选题准确率"""
    correct = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
    acc = correct / len(references) if references else 0
    logger.info(f"单选题准确率: {acc:.4f}")
    return acc

def compute_qa_metrics(predictions, references):
    """计算问答题的评估指标 (ROUGE 和 BLEU)"""
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bleu_scores = bleu.compute(predictions=predictions, references=references)
    logger.info(f"ROUGE 评分: {rouge_scores}")
    logger.info(f"BLEU 评分: {bleu_scores}")
    return {"ROUGE": rouge_scores, "BLEU": bleu_scores}

def call_gpt4_eval(predictions, references, task_type):
    """调用 GPT-4 进行评估"""
    logger.info("正在调用 GPT-4 进行评分...")
    responses = []
    for pred, ref in zip(predictions, references):
        prompt = f"""
        你是一个严格的评分员，请对以下{task_type}任务的回答进行打分（0-100）。
        评分标准：
        1. **正确性**：答案是否符合参考答案。
        2. **完整性**：回答是否覆盖了核心信息。
        3. **表达清晰度**：回答是否流畅易懂。

        **参考答案**: {ref}
        **模型输出**: {pred}

        请按照上述评分标准进行评分，并给出简要解释：
        """
        try:
            score_text = run_chatgpt(query=prompt, model="gpt-4", num_tokens_request=100)
            responses.append(score_text.strip())
        except Exception as e:
            logger.error(f"GPT-4 评分失败: {e}")
            responses.append("评分失败")
    logger.info(f"GPT-4 评分结果: {responses}")
    return responses

def save_results(output_dir, predictions, references, evaluation_results, raw_outputs, original_inputs=None):
    """保存 JSON 和 CSV 评估结果"""
    # ==== 1️⃣ JSON 保存 ====
    predictions_file = os.path.join(output_dir, "predictions.json")
    metrics_file = os.path.join(output_dir, "metrics.json")

    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump({
            "inputs": original_inputs,
            "predictions": predictions,
            "references": references,
            "raw_outputs": raw_outputs
        }, f, ensure_ascii=False, indent=4)
    logger.info(f"🔹 JSON 推理结果已保存至 {predictions_file}")

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
    logger.info(f"🔹 评估指标已保存至 {metrics_file}")

    # ==== 2️⃣ CSV 保存 ====
    csv_file = os.path.join(output_dir, "predictions.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "input", "raw_output", "predicted", "reference", "correct"])
        for idx, (inp, raw, pred, ref) in enumerate(zip(original_inputs, raw_outputs, predictions, references)):
            correct = "✓" if pred == ref else "✗"
            writer.writerow([idx + 1, inp, raw, pred, ref, correct])
    logger.info(f"📄 CSV 推理结果已保存至 {csv_file}")


def generate_text(model, tokenizer, input_texts, max_new_tokens=4096, temperature=0.7):
    """使用 Hugging Face 进行文本生成（支持批量推理，返回原始输出）"""
    inputs = tokenizer(
        input_texts, padding=True, truncation=True, max_length=1024,
        padding_side='left', return_tensors="pt"
    ).to(model.device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_sequences = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )

    new_tokens = output_sequences[:, input_length:]
    output_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    output_texts = [text.strip() for text in output_texts]

    final_answers = []
    for output in output_texts:
        thought_process, answer_candidate = extract_answer_from_tags(output)
        option = extract_first_option(answer_candidate) if not answer_candidate.strip() in ["A", "B", "C", "D"] else answer_candidate.strip()
        if option:
            final_answers.append(option)
        else:
            gpt4_answer = infer_answer_with_gpt4(thought_process, answer_candidate)
            final_answers.append(gpt4_answer)

    return final_answers, output_texts

def evaluate_model(model, tokenizer, dataloader, total_batches, task_type, evaluation_method, output_dir, temperature):
    """使用 Hugging Face 进行评估（支持 PyTorch DataLoader）"""
    logger.info(f"开始评估模型: {task_type}")

    os.makedirs(output_dir, exist_ok=True)
    predictions, references_list, raw_outputs, input_texts = [], [], [], []

    with tqdm(total=total_batches, desc="模型推理中", unit="batch") as pbar:
        # for batch_inputs, batch_references in dataloader:
        #     batch_preds, batch_raw = generate_text(model, tokenizer, batch_inputs, temperature=temperature)
        for batch_items in dataloader:
            prompts = [item["prompt"] for item in batch_items]
            batch_references = [item["answer"] for item in batch_items]

            batch_preds, batch_raw = generate_text(model, tokenizer, prompts, temperature=temperature)

            predictions.extend(batch_preds)
            raw_outputs.extend(batch_raw)
            references_list.extend(batch_references)
            input_texts.extend(prompts)  
            pbar.update(1)

    evaluation_results = {}
    if task_type == "mcq":
        evaluation_results = {"accuracy": compute_mcq_accuracy(predictions, references_list)}
    elif task_type == "qa":
        evaluation_results = (
            compute_qa_metrics(predictions, references_list) if evaluation_method == "metrics"
            else {"GPT-4 Scores": call_gpt4_eval(predictions, references_list, task_type)}
            if evaluation_method == "gpt4"
            else {"predictions": predictions, "references": references_list}
        )

    save_results(output_dir, predictions, references_list, evaluation_results, raw_outputs)
    logger.info(f"评估结果已保存至 {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="大模型评估管线")
    parser.add_argument("--model_path", type=str, default="/mnt/sda/wyp/models/ds-qwen7b-sft", help="本地模型路径")
    parser.add_argument("--input_file", type=str, default="/mnt/sda/wyp/forestllm-main/forest_eval/forest_val_v1.csv", help="输入数据文件")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_data/", help="评估结果存储目录")
    parser.add_argument("--task_type", type=str, choices=["mcq", "qa"], default="mcq", help="任务类型")
    parser.add_argument("--evaluation_method", type=str, choices=["metrics", "gpt4", "manual"], default="metrics", help="评估方式")
    parser.add_argument("--batch_size", type=int, default=2, help="批量推理大小")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.3, help="生成温度")
    parser.add_argument("--model_mode", type=str, default='cot', help="模板选择")
    

    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path, temperature=args.temperature)
    output_dir = os.path.join(args.output_dir, args.model_path.split("/")[-1])
    os.makedirs(output_dir, exist_ok=True)
    print("评估结果将存储在：", output_dir)

    dataloader, total_batches  = get_dataloader(args.input_file, batch_size=args.batch_size, task_type=args.task_type, model_mode=args.model_mode)
    evaluate_model(
        model, tokenizer, dataloader, total_batches, args.task_type, args.evaluation_method, output_dir, args.temperature
    )


if __name__ == "__main__":
    main()
