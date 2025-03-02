import json
import re
from tqdm import tqdm  # 进度条显示，可选

def fix_cot_format(content):
    """ 修复 CoT 格式，去除 'CoT": ' 但保留 `</think>` 之前和之后的所有文本 """

    # **改进正则表达式**
    cot_pattern = r'<think>\s*"CoT":\s*"(.*?)"(.*?)</think>(.*)'

    match = re.search(cot_pattern, content, re.DOTALL)

    if match:
        cot_content = match.group(1).strip()  # `CoT` 主要内容
        after_think = match.group(3).strip()  # `</think>` 之后的内容3

        return f"<think> {cot_content} </think> {after_think}"
    
    # **如果正则匹配失败，使用 find() 进行手动解析**
    # 1️⃣ **找到 `"CoT": "` 位置**
    cot_start = content.find('"CoT": "')
    think_end_idx = content.find("</think>")

    # **如果找不到 `"CoT": "` 或 `</think>`，直接返回原文本**
    if cot_start == -1 or think_end_idx == -1:
        return content.strip()

    # **去掉 `"CoT": "`**
    content = content[cot_start + len('"CoT": "'):]  # 只保留 `CoT` 之后的内容

    # 2️⃣ **找到 `</think>` 之前最近的 `"`**
    last_quote_idx = content.rfind('"', 0, think_end_idx)

    if last_quote_idx != -1:
        last_quote_idx = content.find('"')
        cot_content = content[:last_quote_idx].strip()  # 保留 `CoT` 主要内容
    else:
        cot_content = content[:think_end_idx - len('"CoT": "') - len('</think>') + 1].strip()  # 如果没找到 `"`，保留 `</think>` 之前所有内容

    # 3️⃣ **找到 `</think>` 之后的文本**
    after_think = content[content.find("</think>") + len('</think>'):].strip()

    return f"<think> {cot_content} </think> {after_think}".strip()


def process_jsonl(input_path, output_path):
    """ 处理 JSONL 文件，修复 CoT 格式 """
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line in tqdm(infile):
            try:
                # 解析 JSON 对象
                entry = json.loads(line.strip())

                # 遍历 messages，处理 assistant 的内容
                for msg in entry.get('messages', []):
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        msg['content'] = fix_cot_format(content)

                # 写入处理后的数据
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

            except json.JSONDecodeError:
                print(f"❌ JSON 解析失败的行：{line}")
            except Exception as e:
                print(f"⚠️ 处理异常：{str(e)}")

# 使用示例
process_jsonl('/home/wyp/project/forest/forestllm-main/outputs/sft_data/final/train_data.jsonl', 'output_cleaned.jsonl')
