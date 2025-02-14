import argparse
import json


def add_class_to_jsonl(input_file, output_file, class_name):
    """为每条JSONL数据增加class属性"""
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            data = json.loads(line.strip())
            data["class"] = class_name
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
    print(f"数据已处理完成，输出保存到 {output_file}")


def main():
    parser = argparse.ArgumentParser(description="为JSONL文件增加class属性")
    parser.add_argument(
        "--input",
        default=r"F:\CODE\code\forestInstruct\data\orgs\web_deduped-1114.jsonl",
        help="输入JSONL文件路径",
    )
    parser.add_argument(
        "--output",
        default=r"F:\CODE\code\forestInstruct\data\mateinfo\web_deduped-1114.jsonl",
        help="输出JSONL文件路径",
    )
    parser.add_argument("--class_name", default="web", help="要添加的class属性值")
    args = parser.parse_args()

    add_class_to_jsonl(args.input, args.output, args.class_name)


if __name__ == "__main__":
    main()
