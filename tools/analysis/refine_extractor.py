import json
import re
import argparse
import os
import sys

def extract_snippet_as_string(text):
    """
    从给定字符串中查找被 ```json ... ``` 或 ``` ... ``` 包裹的内容，
    去掉三重反引号后返回纯文本。
    如果未找到匹配，返回 None。
    """
    # 允许 ```json 或者单纯的 ``` 来标记
    pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        snippet = match.group(1)  # 提取到被三重反引号包裹的正文
        # 可以再做 strip 操作，去掉前后空白
        snippet = snippet.strip()
        return snippet
    return None

def replace_refined_response_rawstring(input_file, output_file):
    """
    读取 input_file 中的 JSON 数据:
    - 遍历每个条目下的 refined_questions
    - 如果 requires_refinement == true，且 refined_response 中存在 ``` ... ``` 片段
      则将去掉 ``` 的文本作为字符串写回 refined_response。
    - 最后把修改后的数据写到 output_file。
    """
    # 1. 校验文件是否存在
    if not os.path.isfile(input_file):
        print(f"错误：输入文件 '{input_file}' 不存在。")
        sys.exit(1)

    # 2. 读取原始 JSON 数据
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误：无法解析 JSON 文件。详细信息：{e}")
        sys.exit(1)

    # 假设顶层是 list，根据实际情况修改
    if not isinstance(data, list):
        print("警告：顶层数据不是列表，脚本示例假定顶层是列表。请根据实际结构进行修改。")
        return

    for idx, entry in enumerate(data):
        # 示例：在 entry["expert_agent"]["refined_questions"] 下面
        # 若你的数据结构不同，请修改此处
        if "expert_agent" not in entry:
            continue
        if "refined_questions" not in entry["expert_agent"]:
            continue

        refined_questions = entry["expert_agent"]["refined_questions"]
        if not isinstance(refined_questions, list):
            continue

        # 遍历 refined_questions
        for qidx, qitem in enumerate(refined_questions):
            if not isinstance(qitem, dict):
                continue
            # 1) 检查 requires_refinement 是否为 True
            requires = qitem.get("requires_refinement", False)
            if not requires:
                continue  # 如果不需要 refinement，跳过

            # 2) 获取 refined_response 字段
            refined_response = qitem.get("refined_response", "")
            if not refined_response:
                continue

            # 3) 尝试提取去掉 ```json...``` 的纯文本
            snippet_str = extract_snippet_as_string(refined_response)
            if snippet_str is not None:
                # 找到被三引号包裹的文本，直接把这个纯文本替换回
                qitem["refined_response"] = snippet_str

                print(f"已替换条目ID={entry.get('id', f'unknown_{idx}')}, question_index={qidx} 的 refined_response。")

    # 4. 写回新的 JSON 文件
    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            json.dump(data, out_f, ensure_ascii=False, indent=4)
        print(f"成功：已将修改后的数据写入 '{output_file}'")
    except Exception as e:
        print(f"错误：无法保存数据到文件。详细信息：{e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="当 requires_refinement == true，去除 refined_response 中 ```json ... ``` 标记，保留原始试题格式作为字符串。"
    )
    parser.add_argument('--input_file', default='/home/wyp/project/ForestLLM/outputs/0113/qwen_web_output_deduplicated_ex.json', help='输入的JSON数据文件路径')
    parser.add_argument('--output_file', default='/home/wyp/project/ForestLLM/outputs/0113/qwen_web_output_deduplicated_ex111.json', help='输出的JSON数据文件路径')

    args = parser.parse_args()

    replace_refined_response_rawstring(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
