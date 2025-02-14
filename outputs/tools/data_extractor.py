import json
import argparse
import os
import sys

# 定义属性与步骤的映射关系
ATTRIBUTE_STEP_MAP = {
    'question_setter': '1',
    'expert_agent': '2',
    'virtual_teacher': '3',
    # 如果有更多属性和步骤的对应关系，可以在这里添加
}

def extract_data_from_entry(entry, base_info_keys, keywords, attribute_step_map):
    """
    从单个条目中提取基础信息和指定关键字的数据，同时更新steps字段。

    :param entry: 单个数据条目（字典）
    :param base_info_keys: 基础信息的键列表
    :param keywords: 需要提取的关键字列表
    :param attribute_step_map: 属性与步骤的映射字典
    :return: 提取后的数据字典
    """
    extracted = {key: entry.get(key, None) for key in base_info_keys}

    # 提取指定关键字的数据
    for key in keywords:
        if key in entry:
            extracted[key] = entry[key]
        else:
            print(f"警告：关键字 '{key}' 在条目ID '{entry.get('id', '未知')}' 中不存在。")

    # 更新steps字段
    # 复制原有的steps以避免修改原始数据
    original_steps = entry.get('steps', {})
    updated_steps = original_steps.copy()

    # 遍历所有映射的属性与步骤
    for attr, step in attribute_step_map.items():
        if attr in keywords:
            updated_steps[step] = 'completed'
        else:
            # 仅修改已存在的步骤
            if step in updated_steps:
                updated_steps[step] = 'incomplete'

    extracted['steps'] = updated_steps

    return extracted

def extract_data(input_file, output_file, keywords, attribute_step_map):
    """
    根据指定的关键字从输入的JSON文件中提取数据，并保存到输出文件中，同时更新steps字段。

    :param input_file: 原始JSON文件路径
    :param output_file: 输出JSON文件路径
    :param keywords: 需要提取的关键字列表
    :param attribute_step_map: 属性与步骤的映射字典
    """
    # 检查输入文件是否存在
    if not os.path.isfile(input_file):
        print(f"错误：输入文件 '{input_file}' 不存在。")
        sys.exit(1)

    # 读取原始JSON数据
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误：无法解析JSON文件。详细信息：{e}")
        sys.exit(1)

    # 检查数据是否为列表
    if not isinstance(data, list):
        print("错误：输入的JSON数据应该是一个列表（数组）。")
        sys.exit(1)

    # 定义基础信息键
    base_info_keys = ['id', 'steps', 'class']

    # 提取每个条目的数据
    extracted_data = []
    for entry in data:
        if not isinstance(entry, dict):
            print("警告：发现一个非字典类型的条目，跳过。")
            continue
        extracted_entry = extract_data_from_entry(entry, base_info_keys, keywords, attribute_step_map)
        extracted_data.append(extracted_entry)

    # 保存到输出文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=4)
        print(f"成功：数据已保存到 '{output_file}'")
    except Exception as e:
        print(f"错误：无法保存数据到文件。详细信息：{e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="数据处理工具：根据关键字提取JSON数据部分。支持多条目数据，并更新steps字段。")
    parser.add_argument('--input_file', default='/home/wyp/project/ForestLLM/outputs/0113/qwen_web_output_deduplicated.json', help='输入的JSON数据文件路径')
    parser.add_argument('--output_file', default='/home/wyp/project/ForestLLM/outputs/0113/qwen_web_output_deduplicated_ex.json', help='输出的JSON数据文件路径')
    parser.add_argument('--keywords', default=['question_setter', 'expert_agent'], nargs='+', help='需要提取的关键字，例如：question_setter')

    args = parser.parse_args()

    # 检查关键字是否在映射关系中
    invalid_keywords = [kw for kw in args.keywords if kw not in ATTRIBUTE_STEP_MAP]
    if invalid_keywords:
        print(f"错误：以下关键字未在属性与步骤的映射关系中定义：{', '.join(invalid_keywords)}")
        sys.exit(1)

    extract_data(args.input_file, args.output_file, args.keywords, ATTRIBUTE_STEP_MAP)

if __name__ == "__main__":
    main()

