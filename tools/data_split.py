import pandas as pd
from sklearn.model_selection import train_test_split

# 读取 JSONL 文件
input_file = '/home/wyp/project/forest/forestllm-main/outputs/sft_data/book/train_multiple_choice.jsonl'
data = pd.read_json(input_file, lines=True)

# 检查数据是否包含 'id' 和 '知识点' 列
if 'id' not in data.columns or '知识点' not in data.columns:
    raise ValueError("数据中缺少 'id' 或 '知识点' 列")

# 按照 80% 训练集和 20% 测试集的比例划分数据
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 保存划分后的数据集
train_output_file = '/home/wyp/project/forest/forestllm-main/outputs/sft_data/book/train_multiple_choice_train.jsonl'
test_output_file = '/home/wyp/project/forest/forestllm-main/outputs/sft_data/book/train_multiple_choice_test.jsonl'

train_data.to_json(train_output_file, orient='records', lines=True, force_ascii=False)
test_data.to_json(test_output_file, orient='records', lines=True, force_ascii=False)

print(f"训练集已保存至: {train_output_file}")
print(f"测试集已保存至: {test_output_file}")
