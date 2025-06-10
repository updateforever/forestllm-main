import os
import json
import argparse
import logging
from tqdm import tqdm
from openai import OpenAI
from datetime import datetime
from data.dataset import get_dataloader
from eval import (
    extract_answer_from_tags, extract_first_option, infer_answer_with_gpt4
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_existing_ids(save_file):
    if not os.path.exists(save_file): return set()
    with open(save_file, 'r', encoding='utf-8') as f:
        return set(int(json.loads(line)["id"]) for line in f if line.strip())


def call_api_batch(messages_list, model="Qwen2.5-7B-Instruct", base_url="http://127.0.0.1:8000/v1", api_key="EMPTY", temperature=0.3, max_tokens=512):
    # client = OpenAI(api_key=api_key, base_url=base_url)
    client = OpenAI(
        api_key="sk-9fca3e0e00994b96835cf550bb254ba0",  # 使用您的 Dashscope API 密钥
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    responses = []
    for messages in messages_list:
        resp = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        output = resp.choices[0].message.content.strip()
        responses.append(output)
    return responses


def evaluate_api_model(dataloader, total_batches, args, save_path, finished_ids):
    """
    使用 API 批量推理并保存结果到 JSONL，支持断点恢复。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    logger.info(f"🔹 正在保存到: {save_path}")

    with open(save_path, "a", encoding="utf-8") as fout, tqdm(desc="API 推理中", unit="sample") as pbar:
        for batch in dataloader:
            batch_prompts = [item["prompt"] for item in batch]
            batch_messages = [item["messages"] for item in batch]
            batch_answers = [item["answer"] for item in batch]
            batch_ids = [item["id"] for item in batch]

            real_batch_messages = []
            real_batch_infos = []

            for prompt, messages, ref, id_ in zip(batch_prompts, batch_messages, batch_answers, batch_ids):
                if id_ in finished_ids:
                    continue
                real_batch_messages.append(messages)  # ✅ 用 messages 作为 API 输入
                real_batch_infos.append((id_, prompt, messages, ref))

            if not real_batch_messages:
                continue  # 这个 batch 全部跳过了

            # 🔥✅ 批量调用 API（用封装好的 call_api_batch）
            responses = call_api_batch(
                messages_list=real_batch_messages,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                temperature=args.temperature,
                max_tokens=512
            )

            for (id_, prompt, messages, ref), output in zip(real_batch_infos, responses):
                thought, candidate = extract_answer_from_tags(output)
                candidate = candidate.strip().upper()

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
                    "message": messages,
                    "reference": ref,
                    "raw_output": output,
                    "predicted": pred,
                    "correct": pred == ref,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

            pbar.update(len(real_batch_messages))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="/mnt/sda/wyp/forestllm-main/forest_eval/forest_zero_shot.csv")
    parser.add_argument("--output_dir", type=str, default="/mnt/sda/wyp/forestllm-main/outputs/eval_api")
    parser.add_argument("--task_type", choices=["mcq", "qa"], default="mcq")
    parser.add_argument("--evaluation_method", choices=["metrics", "gpt4", "manual"], default="metrics")
    parser.add_argument("--model", type=str, default="gpt4")
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--model_mode", type=str, default='normal')
    args = parser.parse_args()

    eval_split = os.path.splitext(os.path.basename(args.input_file))[0]
    save_root = os.path.join(args.output_dir, args.model, eval_split)

    os.makedirs(save_root, exist_ok=True)

    save_file = os.path.join(save_root, "results.jsonl")

    done_ids = load_existing_ids(save_file)
    logger.info(f"🔹 已完成 {len(done_ids)} 条，将跳过这些样本...")

    dataloader, total_batches = get_dataloader(
        args.input_file,
        batch_size=args.batch_size,
        task_type=args.task_type,
        model_mode=args.model_mode
    )

    evaluate_api_model(dataloader, total_batches, args, save_file, done_ids)
    logger.info("✅ 推理任务已完成。")


if __name__ == "__main__":
    main()
