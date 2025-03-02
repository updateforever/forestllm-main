import re
import json
import docx
import pdfplumber
from pathlib import Path

def extract_text(file_path):
    """通用文本提取函数"""
    path = Path(file_path)
    
    if path.suffix.lower() == '.pdf':
        return extract_pdf_text(path)
    elif path.suffix.lower() in ('.docx', '.doc'):
        return extract_docx_text(path)
    else:
        raise ValueError("不支持的文件格式")

def extract_pdf_text(file_path):
    """提取PDF文本（处理简单版式）"""
    full_text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if text:
                full_text.append(text)
    return "\n".join(full_text)

def extract_docx_text(file_path):
    """提取Word文档文本"""
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def clean_text(content):
    """深度文本清洗"""
    content = re.sub(r'[\x00-\x1F\\]', '', content)  # 移除控制字符
    content = re.sub(r'\n{3,}', '\n\n', content)  # 合并多余换行
    content = re.sub(r'第[一二三四五六七八九十]+节\s*', '', content)  # 移除页眉页脚
    return content.strip()

def parse_questions(content):
    """智能题目解析"""
    questions = []
    
    # 题目分割：匹配编号、题目和选项
    question_blocks = re.findall(r'(\d+)[\.、．]\s*(.+?)(?=\n\d+[\.、．]|\Z)', content, re.DOTALL)

    for q_num, block in question_blocks:
        try:
            q_num = int(q_num)

            # 提取答案
            answer_match = re.search(r'【答案】\s*([A-D]+)', block)
            if not answer_match:
                continue
            answer = answer_match.group(1).upper()
            block = block[:answer_match.start()]  # 删除答案部分

            # 题目与选项分离
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            question_body = []
            options = []
            
            option_pattern = re.compile(r'^([A-D])[\.．、]?\s*(.*)$')
            for line in lines:
                if match := option_pattern.match(line):
                    options.append({
                        "option_id": match.group(1),
                        "content": match.group(2)
                    })
                else:
                    question_body.append(line)
            
            if not options:
                continue  # 确保没有选项的题目不会出错
            
            questions.append({
                "id": q_num,
                "question": "\n".join(question_body),
                "options": options,
                "answer": answer,
                "type": "single_choice" if len(answer) == 1 else "multiple_choice",
                "category": "森林消防知识",
                "difficulty": "待定"
            })
        
        except Exception as e:
            print(f"解析第{q_num}题时出错: {str(e)}")
            continue
    
    return questions

def convert_to_benchmark(input_path, output_path):
    """执行完整转换流程"""
    try:
        print(f"正在处理文件: {input_path}")
        raw_text = extract_text(input_path)
        cleaned_text = clean_text(raw_text)
        questions = parse_questions(cleaned_text)
        
        dataset = {
            "version": "2.0",
            "dataset_info": {
                "source_type": Path(input_path).suffix[1:].upper(),
                "total_questions": len(questions)
            },
            "questions": questions
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
            
        print(f"成功转换{len(questions)}道题目到 {output_path}")
        
    except Exception as e:
        print(f"转换失败: {str(e)}")

if __name__ == '__main__':
    input_file = "benchmark/应急管理局森林草原消防队员招聘考试《森林消防基础知识》真题精选汇总及答案_2288.pdf"
    output_file = "benchmark/exam_benchmark.json"
    convert_to_benchmark(input_file, output_file)
