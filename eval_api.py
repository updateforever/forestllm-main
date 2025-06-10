import os
import json
import argparse
import logging
from tqdm import tqdm
from openai import OpenAI
from datetime import datetime
from data.dataset import get_dataloader
from eval import (
    extract_answer_from_tags, extract_first_option,
    is_valid_option, compute_mcq_accuracy,
    compute_qa_metrics, call_gpt4_eval, infer_answer_with_gpt4
)

# === 日志配置 ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def call_api_batch(messages_list, model="Qwen2.5-7B-Instruct", base_url="http://127.0.0.1:8000/v1", api_key="EMPTY", temperature=0.3, max_tokens=512):
    client = OpenAI(api_key=api_key, base_url=base_url)
    responses = []
    for messages in messages_list:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        output = resp.choices[0].message.content.strip()
        responses.append(output)
    return responses

def evaluate_api_model(data_loader, total_batches, args):
    predictions, references_list, prompts = [], [], []

    with tqdm(total=total_batches, desc="API 推理中") as pbar:
        for batch in data_loader:
            batch_messages = [item["messages"] for item in batch]
            references = [item["answer"] for item in batch]
            prompts.extend([item["prompt"] for item in batch])

            responses = call_api_batch(
                batch_messages,
                model=args.model,
                base_url=args.base_url,
                temperature=args.temperature,
                api_key=args.api_key
            )

            for output, reference in zip(responses, references):
                thought, candidate = extract_answer_from_tags(output)
                # 先标准化一下原始回答内容
                candidate = candidate.strip().upper()

                # 如果 candidate 是合法的单个选项（比如 "A", "B", "C", "D"）
                if candidate in ["A", "B", "C", "D"]:
                    option = candidate
                # 否则，尝试从更复杂的内容中提取第一个选项
                else:
                    option = extract_first_option(candidate)

                if option:
                    # 如果成功提取了合法选项（A/B/C/D），直接使用
                    predictions.append(option)
                else:
                    # 否则调用 GPT-4 进行推理判断，并加入预测结果
                    corrected_answer = infer_answer_with_gpt4(candidate)
                    predictions.append(corrected_answer)

                references_list.append(reference)

            pbar.update(1)

    return predictions, references_list, prompts

def save_eval_results(output_dir, prompts, predictions, references, metrics):
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, "results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "prompts": prompts,
            "predictions": predictions,
            "references": references,
            "metrics": metrics
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 推理与评估结果已保存至: {result_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="/mnt/sda/wyp/forestllm-main/forest_eval/book/eval_multiple_choice_filtered.csv")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_api")
    parser.add_argument("--task_type", choices=["mcq", "qa"], default="mcq")
    parser.add_argument("--evaluation_method", choices=["metrics", "gpt4", "manual"], default="metrics")
    parser.add_argument("--model", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--model_mode", type=str, default='cot', help="模板选择")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 📁 提取数据集名（不带路径和后缀）
    eval_split = os.path.basename(args.input_file).split('.')[0]
    # 🗂️ 构建保存路径
    save_dir = os.path.join(args.output_dir, args.model, eval_split, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"📂 保存路径：{save_dir}")

    dataloader, total_batches = get_dataloader(
        args.input_file,
        batch_size=args.batch_size,
        task_type=args.task_type,
        model_mode=args.model_mode
    )
    predictions, references, prompts = evaluate_api_model(dataloader, total_batches, args)

    if args.task_type == "mcq":
        metrics = {"accuracy": compute_mcq_accuracy(predictions, references)}
    elif args.task_type == "qa":
        if args.evaluation_method == "metrics":
            metrics = compute_qa_metrics(predictions, references)
        elif args.evaluation_method == "gpt4":
            metrics = {"GPT-4": call_gpt4_eval(predictions, references, task_type="qa")}
        else:
            metrics = {}

    save_eval_results(save_dir, prompts, predictions, references, metrics)

if __name__ == "__main__":
    main()


# o1 sk-QkNZIBxy3lD2h7H95CH0wIXEnXevpoEvWHhVGsHGSBqwGTCj