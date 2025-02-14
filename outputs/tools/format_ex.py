#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os


def transform_data(data_input):
    """
    将原始数据进行转换：对每条数据内的 question_setter.questions
    分别生成单独的记录，并将其打包在 "question_setter" 字段中，同时合并
    expert_agent.refined_questions[i] & virtual_teacher.processed_results[i]
    """
    output = []

    for entry in data_input:
        entry_id = entry.get("id", "")
        entry_steps = entry.get("steps", {})
        entry_class = entry.get("class", "")

        # 取出问题列表
        questions = entry.get("question_setter", {}).get("questions", [])

        # 取出 expert_agent, virtual_teacher
        refined_questions = entry.get("expert_agent", {}).get("refined_questions", [])
        processed_results = entry.get("virtual_teacher", {}).get(
            "processed_results", []
        )

        # 遍历每个问题
        for i, q in enumerate(questions):
            new_record = {
                "id": entry_id,
                "steps": entry_steps,
                "class": entry_class,
                "question_setter": {
                    "knowledge": q.get("knowledge", ""),
                    "difficulty": q.get("difficulty", ""),
                    "question_type": q.get("question_type", ""),
                    "response": q.get("response", ""),
                },
                # 保留同索引下的 expert_agent
                "expert_agent": (
                    refined_questions[i] if i < len(refined_questions) else {}
                ),
                # 保留同索引下的 virtual_teacher
                "virtual_teacher": (
                    processed_results[i] if i < len(processed_results) else {}
                ),
            }
            output.append(new_record)

    return output


def main():
    # 输入文件路径
    input_path = "/home/wyp/project/ForestLLM/outputs/article/qwen_article_output_02_lowrelevance_filtered.json"

    # 输出文件路径（同目录下，文件名根据需要自行修改）
    output_path = "/home/wyp/project/ForestLLM/outputs/article/qwen_article_output_03_transformed.json"

    # 读取输入文件
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 执行转换
    transformed_data = transform_data(data)

    # 将转换结果写入到输出文件
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(transformed_data, out_f, ensure_ascii=False, indent=4)

    print(f"转换完成！结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
