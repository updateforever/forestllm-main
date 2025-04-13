# finl_eval_prompt.py

# ====== Eval Prompts ======

GRADE_PROMPT_EN = """
You are a forestry evaluation expert. \
Review and grade the following content for its suitability in a forestry instruction-tuning dataset.\
Evaluate the original question, expert refinements, conversational format, and student's answer.\
Provide a score (1-10) and concise feedback on the overall quality and alignment with the reference\
answer.

Background: '{text}'
Original Response: '{original_response}'
Refined Response: '{refined_response}'
Conversational Form: '{conversational_form}'
Student's Answer: '{student_answer}'

Output format:
{{
    "score": "<score>", 
    "feedback": "<feedback>"
}}
"""

GRADE_PROMPT_CN1 = """
你是一位经验丰富的评估专家，专注于分析学生的作答情况以判断当前试题的重要性。

请基于以下提供的原始试题、参考答案以及学生作答进行全面分析，着重关注以下方面：
    1.理解程度：学生对问题的理解是否到位？回答是否偏离问题本身？
    2.准确性： 回答是否与标准答案保持一致？

重要性评估标准：
    1.如果学生的回答存在明显的理解偏差、逻辑错误或较大的知识空白，请将该条数据标记为高重要性。
    2.如果学生的回答基本正确，但在细节上有所欠缺，请标记为中等重要性。
    3.如果学生的回答完全正确，且逻辑清晰，请标记为低重要性。

请严格按照以下 JSON 格式输出分析结果，不要包含多余的文本或格式：
{{
  "score": "<1-10的分数，1为非常差，10为非常好>",
  "importance_level": "<h/m/l 分别代表高/中/低重要性>",
  "feedback": "<简洁明了的反馈，描述学生的表现>"
}}

原始试题和参考答案：'{response}'
学生作答：'{student_answer}'
"""

GRADE_PROMPT_CN = """
你是一位经验丰富的评估专家，专注于分析学生的作答情况并判断当前试题的掌握程度。

在评分时，需要从以下三个主要维度进行综合判断：
1. 理解程度（Understanding）：学生对题目的理解是否准确，是否存在明显的偏离或误解？
2. 准确性（Accuracy）：学生的回答与参考答案的匹配度如何？是否展现了完整、清晰的思路与证据？
3. 回答通顺度（Fluency）：学生的回答是否通顺、流畅，是否存在明显的语言或逻辑矛盾、重复或“幻觉”信息？

请根据上述维度为学生的整体表现给出一个 **1-10** 的分数（score）：
- **1-2 分**：回答与题意严重偏离或逻辑混乱，几乎无法体现对题目要求的理解，且表达极不通顺。
- **3-5 分**：部分内容正确，但存在明显错误或理解偏差，表达欠缺流畅性，可能出现多处重复或不合理信息。
- **6-8 分**：回答基本正确，思路较为清晰，表达整体通顺，但在细节或逻辑上仍有改进空间。
- **9-10 分**：回答高度准确、结构合理且表达流畅，能全面把握问题并无明显多余或幻觉内容。

同时，为了表征学生对该知识点的**掌握程度（mastery_level）**，请依照以下原则做出标记：
1. 如果学生的回答存在明显的理解偏差或重大错误，说明对知识点的掌握程度较低，请标记为 **“l”**（低掌握）。
2. 如果学生的回答基本正确，但在细节或思路上有所欠缺，说明对知识点的掌握程度一般，请标记为 **“m”**（中等掌握）。
3. 如果学生的回答完全正确、思路清晰流畅，说明对知识点的掌握程度较高，请标记为 **“h”**（高掌握）。

请基于以下提供的原始试题、参考答案以及学生作答进行全面分析:
原始试题和参考答案：'{response}'
学生作答：'{student_answer}'

请严格按照以下 JSON 格式输出分析结果，不要包含多余的文本或格式：
{{
  "score": "<1-10的分数，1为非常差，10为非常好>",
  "mastery_level": "<h/m/l 分别代表高/中/低掌握>"
}}
"""

GRADE_PROMPT_CN2 = """
你是一位林业专家，请你对以下三名学生林业试题的作答情况进行多维度评估。
对于每个回答，请根据下述规则打分：  
1. mastery_score（1-5分）：关注回答是否正确。主观题的输出越接近正确答案代表知识点掌握度较高，选择题的输出只要包含正确选项字母则代表知识点掌握度较高。  
2. accuracy_score（1-5分）：关注回答与参考答案的符合程度、是否有错误/幻觉信息。  
3. fluency_score（1-5分）：关注回答是否通顺、连贯、无重复或矛盾表述。  

如果某个回答是空字符串 ""（表示未作答），则将该回答的评估项都设置为null。  
请对以下数据进行评估：  
- 题目及参考答案：{response}  
- 学生作答列表：{student_answer}
请严格按照以下 JSON 语法格式返回数据（确保 none 用 "none"（字符串）表示）,无需输出其它内容：  
[
  {{
    "id": "0/1/2",
    "mastery_score": "x",
    "accuracy_score": "y",
    "fluency_score": "z"
  }},
  ...
]
"""

GRADE_PROMPT_CN_FINAL = """
# 🎯 角色设定
你是一位严谨的林业专家，负责对学生林业试题作答情况进行全面评估。你的评估结果将用于清洗用于大模型训练的微调数据，请务必专业、客观、标准化。

# 📌 评估说明
请你参考以下题目信息与标准答案，并对三位学生的作答进行逐项打分，评分维度如下：

1. mastery_score（1-5分）：判断学生是否掌握了对应知识点。  
   - 主观题：回答是否贴合核心内容，结构清晰、逻辑严谨。
   - 选择题：只要包含正确选项（如 A、B 等）即视为基本掌握。

2. accuracy_score（1-5分）：评估回答的准确性，是否存在事实性错误或幻觉内容，是否与参考答案一致。

3. fluency_score（1-5分）：语言是否通顺、连贯，有无重复或语病、矛盾语句。
## 约束条件（重要！）
❌ 禁止生成解析等其它无关
⚠️ 如果某个作答为空（即为 `""`），请将该学生的所有评分字段设置为 `"none"`（字符串形式）。

# 📚 输入信息
- 题目与参考答案：（单选题为csv格式，即eval_id,question,A,B,C,D,answer格式）

{response}

- 三位学生的回答列表：
{student_answer}

# ✅ 输出格式（严格遵守以下 JSON 格式，不要输出多余说明文字）
[
  {{
    "id": "0",
    "mastery_score": "x",
    "accuracy_score": "y",
    "fluency_score": "z"
  }},
  ...
]
"""