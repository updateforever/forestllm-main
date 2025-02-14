import json
from collections import defaultdict

# 文件路径
file_path = "/home/wyp/project/ForestLLM/outputs/article/qwen_article_output.json"
output_file_path = "/home/wyp/project/ForestLLM/outputs/article/qwen_article_output_01_deduplicated.json"


def remove_duplicate_ids(file_path, output_file_path):
    try:
        # 加载数据
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # 提取所有的 ID 和对应数据
        id_map = defaultdict(list)
        for entry in data:
            if "id" in entry:
                id_map[entry["id"]].append(entry)

        # 去重逻辑
        deduplicated_data = []
        for id_, entries in id_map.items():
            if len(entries) > 1:
                # 使用第一条作为参考，检查数据是否一致
                reference_entry = json.dumps(entries[0], sort_keys=True)
                all_consistent = all(
                    json.dumps(entry, sort_keys=True) == reference_entry
                    for entry in entries
                )

                if all_consistent:
                    # 如果一致，只保留一个
                    deduplicated_data.append(entries[0])
                else:
                    # 如果不一致，保留所有条目
                    # deduplicated_data.extend(entries)
                    deduplicated_data.append(entries[0])
            else:
                # 非重复数据直接添加
                deduplicated_data.append(entries[0])

        # 写入去重后的数据到新文件
        with open(output_file_path, "w", encoding="utf-8") as outfile:
            json.dump(deduplicated_data, outfile, ensure_ascii=False, indent=4)

        print(
            f"去重完成！总计保留 {len(deduplicated_data)} 条数据。结果已保存到 {output_file_path}"
        )
    except Exception as e:
        print(f"发生错误: {e}")


# 执行去重
remove_duplicate_ids(file_path, output_file_path)
