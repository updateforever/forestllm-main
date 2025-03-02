# import json
# import os
# import random
# from collections import defaultdict

# def load_data(file_path):
#     """从 JSONL 文件中加载数据"""
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             data.append(json.loads(line.strip()))
#     return data

# def save_data(data, file_path):
#     """将数据保存为 JSONL 格式"""
#     with open(file_path, 'w', encoding='utf-8') as f:
#         for item in data:
#             f.write(json.dumps(item, ensure_ascii=False) + '\n')

# def save_id_knowledge_mapping(data, file_path):
#     """保存 id 和 knowledge 的对应关系"""
#     id_knowledge_mapping = {}
#     for item in data:
#         item_id = item.get('id')
#         knowledge_point = item.get('question_setter', {}).get('knowledge', '未知知识点')
#         id_knowledge_mapping[item_id] = knowledge_point
    
#     with open(file_path, 'w', encoding='utf-8') as f:
#         json.dump(id_knowledge_mapping, f, ensure_ascii=False, indent=4)

# def split_data_by_id(data, train_ratio=0.8, seed=None):
#     """
#     根据每个 id 的子数据，将其按比例划分为训练集和测试集
#     """
#     if seed is not None:
#         random.seed(seed)
    
#     # 按 id 分组数据
#     data_by_id = defaultdict(list)
#     for item in data:
#         data_by_id[item['id']].append(item)
    
#     train_data = []
#     test_data = []
    
#     for id, items in data_by_id.items():
#         random.shuffle(items)
#         split_point = int(len(items) * train_ratio)
#         train_data.extend(items[:split_point])
#         test_data.extend(items[split_point:])
    
#     return train_data, test_data

# def main(input_file, train_output_file, test_output_file, id_knowledge_mapping_train_file, id_knowledge_mapping_test_file, train_ratio=0.8):
#     # 加载数据
#     data = load_data(input_file)
    
#     # 按 id 划分数据集
#     train_data, test_data = split_data_by_id(data, train_ratio=train_ratio)
    
#     # 保存数据
#     save_data(train_data, train_output_file)
#     save_data(test_data, test_output_file)
    
#     # 保存 id 和知识点的对应关系
#     save_id_knowledge_mapping(train_data, id_knowledge_mapping_train_file)
#     save_id_knowledge_mapping(test_data, id_knowledge_mapping_test_file)
    
#     print(f"训练集大小: {len(train_data)}")
#     print(f"测试集大小: {len(test_data)}")
#     print(f"训练集的 id 和知识点映射已保存至: {id_knowledge_mapping_train_file}")
#     print(f"测试集的 id 和知识点映射已保存至: {id_knowledge_mapping_test_file}")

# if __name__ == "__main__":
#     input_file = "/home/wyp/project/forest/forestllm-main/outputs/sft_data/final/train_data.jsonl"  # 所有问答数据   多选题也是问答形式    评估时多选题用csv文件吧
#     train_output_file = "/home/wyp/project/forest/forestllm-main/outputs/sft_data/final/train_split.jsonl"
#     test_output_file = "/home/wyp/project/forest/forestllm-main/outputs/sft_data/final/test_split.jsonl"
#     id_knowledge_mapping_train_file = "/home/wyp/project/forest/forestllm-main/outputs/sft_data/final/id_knowledge_mapping_train.json"
#     id_knowledge_mapping_test_file = "/home/wyp/project/forest/forestllm-main/outputs/sft_data/final/id_knowledge_mapping_test.json"

#     main(input_file, train_output_file, test_output_file, id_knowledge_mapping_train_file, id_knowledge_mapping_test_file)


import json
from collections import defaultdict

def load_data(file_path):
    """从 JSONL 文件中加载数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_data(data, file_path):
    """将数据保存为 JSONL 格式"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def save_id_knowledge_mapping(data, file_path):
    """
    保存 id 和 knowledge 的对应关系，确保同一个 id 可能对应多个知识点。
    """
    id_knowledge_mapping = defaultdict(set)  # 使用 set 存储多个知识点，防止重复

    for item in data:
        item_id = item.get('id')
        knowledge_point = item.get('knowledge', '未知知识点')
        
        # 将知识点加入 set
        id_knowledge_mapping[item_id].add(knowledge_point)
    
    # 转换为普通 dict，并将 set 转换为 list，确保 JSON 格式正确
    id_knowledge_mapping = {key: list(value) for key, value in id_knowledge_mapping.items()}

    # 保存到 JSON 文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(id_knowledge_mapping, f, ensure_ascii=False, indent=4)

    print(f"id 和知识点映射已保存至: {file_path}")

def split_data_by_mastery_level(data):
    """
    根据 mastery_level 划分数据：
    - 'l' 的数据为测试集
    - 其他 mastery_level 的数据为训练集
    """
    train_data = []
    test_data = []
    
    for item in data:
        mastery_level = item.get('mastery_level', '')
        
        # mastery_level 为 'l' 的数据全部划分为测试集
        if mastery_level == "l":
            test_data.append(item)
        else:
            train_data.append(item)
    
    return train_data, test_data

def main(input_file, test_output_file, id_knowledge_mapping_test_file):
    # 加载数据
    data = load_data(input_file)
    
    # 按 mastery_level 划分数据集
    _, test_data = split_data_by_mastery_level(data)
    
    # 保存测试集数据
    save_data(test_data, test_output_file)
    
    # 保存 id 和知识点的对应关系
    save_id_knowledge_mapping(test_data, id_knowledge_mapping_test_file)
    
    print(f"测试集大小: {len(test_data)}")
    print(f"测试集的 id 和知识点映射已保存至: {id_knowledge_mapping_test_file}")

if __name__ == "__main__":
    input_file = "/home/wyp/project/forest/forestllm-main/outputs/sft_data/final/train_data.jsonl"  # 所有问答数据  
    test_output_file = "/home/wyp/project/forest/forestllm-main/outputs/sft_data/final/test_split.jsonl"
    id_knowledge_mapping_test_file = "/home/wyp/project/forest/forestllm-main/outputs/sft_data/final/id_knowledge_mapping_test.json"

    main(input_file, test_output_file, id_knowledge_mapping_test_file)
