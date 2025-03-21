# import os
# import json

# import re
# from typing import List, Dict

# def extract_outline(text: str) -> List[Dict]:
#     """
#     提取Markdown文档的大纲
#     :param text: Markdown文本
#     :return: 提取的大纲数组
#     """
#     outline_regex = re.compile(r'^(#{1,6})\s+(.+?)(?:\s*\{#[\w-]+\})?\s*$', re.MULTILINE)
#     outline = []
    
#     for match in outline_regex.finditer(text):
#         level = len(match.group(1))
#         title = match.group(2).strip()
#         outline.append({
#             'level': level,
#             'title': title,
#             'position': match.start()
#         })
    
#     return outline

# def split_by_headings(text: str, outline: List[Dict]) -> List[Dict]:
#     """
#     根据标题分割Markdown文档
#     :param text: Markdown文本
#     :param outline: 文档大纲
#     :return: 按标题分割的段落数组
#     """
#     if not outline:
#         return [{'heading': None, 'level': 0, 'content': text, 'position': 0}]
    
#     sections = []
    
#     if outline[0]['position'] > 0:
#         front_matter = text[:outline[0]['position']].strip()
#         if front_matter:
#             sections.append({'heading': None, 'level': 0, 'content': front_matter, 'position': 0})
    
#     for i, current in enumerate(outline):
#         next_position = outline[i + 1]['position'] if i + 1 < len(outline) else len(text)
#         heading_line = text[current['position']:].split('\n', 1)[0]
#         start_pos = current['position'] + len(heading_line) + 1
#         content = text[start_pos:next_position].strip()
        
#         sections.append({'heading': current['title'], 'level': current['level'], 'content': content, 'position': current['position']})
    
#     return sections

# def split_long_section(section: Dict, max_length: int) -> List[str]:
#     """
#     分割超长段落
#     :param section: 段落对象
#     :param max_length: 最大分割字数
#     :return: 分割后的段落数组
#     """
#     content = section['content']
#     paragraphs = re.split(r'\n\n+', content)
#     result, current_chunk = [], ""

#     for paragraph in paragraphs:
#         if len(paragraph) > max_length:
#             if current_chunk:
#                 result.append(current_chunk)
#                 current_chunk = ""

#             sentences = re.findall(r'[^.!?]+[.!?]+', paragraph) or [paragraph]
#             sentence_chunk = ""

#             for sentence in sentences:
#                 if len(sentence_chunk) + len(sentence) <= max_length:
#                     sentence_chunk += sentence
#                 else:
#                     if sentence_chunk:
#                         result.append(sentence_chunk)
#                     sentence_chunk = sentence if len(sentence) <= max_length else sentence[:max_length]
            
#             if sentence_chunk:
#                 current_chunk = sentence_chunk
#         elif len(current_chunk) + len(paragraph) + 2 <= max_length:
#             current_chunk += f"\n\n{paragraph}" if current_chunk else paragraph
#         else:
#             result.append(current_chunk)
#             current_chunk = paragraph

#     if current_chunk:
#         result.append(current_chunk)

#     return result

# def process_sections(sections: List[Dict], min_length: int, max_length: int) -> List[Dict]:
#     """
#     处理段落，根据最小和最大分割字数进行分割
#     :param sections: 段落数组
#     :param min_length: 最小分割字数
#     :param max_length: 最大分割字数
#     :return: 处理后的段落数组
#     """
#     processed_sections = []
#     accumulated_section = None

#     for section in sections:
#         content_length = len(section['content'].strip())

#         if content_length < min_length:
#             if accumulated_section:
#                 accumulated_section['content'] += f"\n\n{section['content']}"
#             else:
#                 accumulated_section = section.copy()
#         else:
#             if accumulated_section:
#                 processed_sections.append(accumulated_section)
#                 accumulated_section = None
            
#             if content_length > max_length:
#                 split_content = split_long_section(section, max_length)
#                 processed_sections.extend({'summary': f"Part {i+1}/{len(split_content)}", 'content': sub} for i, sub in enumerate(split_content))
#             else:
#                 processed_sections.append({'summary': section.get('heading', 'No heading'), 'content': section['content']})

#     if accumulated_section:
#         processed_sections.append(accumulated_section)

#     return processed_sections

# def split_markdown(markdown_text: str, min_length: int, max_length: int) -> List[Dict]:
#     """
#     拆分Markdown文档
#     :param markdown_text: Markdown文本
#     :param min_length: 最小分割字数
#     :param max_length: 最大分割字数
#     :return: 分割结果数组
#     """
#     outline = extract_outline(markdown_text)
#     sections = split_by_headings(markdown_text, outline)
#     processed_sections = process_sections(sections, min_length, max_length)
    
#     return [{'summary': sec.get('summary', 'None'), 'content': sec['content']} for sec in processed_sections]



# def process_books(input_folder, output_json, min_length=2000, max_length=4000):
#     """处理文件夹内的所有 md 书籍文件，将内容拆分后存入 JSON"""
#     book_data = []

#     # 遍历文件夹
#     for root, _, files in os.walk(input_folder):
#         for file in files:
#             if file.endswith(".md"):  # 只处理 Markdown 文件
#                 book_name = os.path.splitext(file)[0]  # 书名 (去掉.md)
#                 file_path = os.path.join(root, file)

#                 # 读取 .md 文件内容
#                 with open(file_path, "r", encoding="utf-8") as f:
#                     content = f.read().strip()

#                 # 按 length 拆分文本
#                 content_parts = split_markdown(content, min_length, max_length)

#                 # 生成 JSON 结构
#                 for part in content_parts:
#                     book_data.append({
#                         "text": part["content"],
#                         "source": book_name,
#                         "data_class": "book",
#                         "summary": part["summary"]
#                     })

#     # 保存 JSON 文件
#     with open(output_json, "w", encoding="utf-8") as json_file:
#         json.dump(book_data, json_file, ensure_ascii=False, indent=4)

#     print(f"处理完成，数据已保存至 {output_json}")

# # 示例调用
# input_folder = "/home/wyp/project/marker/outputs"  # 你的书籍目录
# output_json = "mateinfo/all_book_data_from_markdown.json"  # 输出 JSON 文件
# process_books(input_folder, output_json)



import json

def convert_json_to_jsonl(json_file, jsonl_file):
    """
    将 JSON 文件转换为 JSONL 文件，每行包含一条 JSON 记录
    :param json_file: 输入的 JSON 文件路径
    :param jsonl_file: 输出的 JSONL 文件路径
    """
    # 读取 JSON 文件
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 处理并转换格式
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for entry in data:
            jsonl_entry = {
                "text": entry["text"],
                "meta_info": {
                    "from": entry["source"],
                    "data_class": entry["data_class"]
                }
            }
            f.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")

    print(f"转换完成，JSONL 文件已保存至 {jsonl_file}")

# 示例调用
convert_json_to_jsonl("/home/wyp/project/forest/forestllm-main/mateinfo/all_book_data_from_markdown.json", "/home/wyp/project/forest/forestllm-main/mateinfo/all_book.jsonl")
