# question_prompts.py

# ====== Knowledge Extraction Prompts ======


KNOWLEDGE_EXTRACTION_BOOK_EN = """
You are a forestry expert tasked with preparing teaching materials. \
   Extract primary knowledge points from the following book content page to aid in educational assessment \
   (e.g., for test creation). Use the main theme and supporting details in the content to summarize broader,\
   more inclusive knowledge points, avoiding overly fragmented points. 
For each point, assess its complexity level as 'simple', 'medium', or 'complex' based on factors like\
   abstractness, applicability, or difficulty of comprehension. 
For example, basic definitions may be considered 'simple', while analyses of ecosystem interactions might\
   be 'complex'.

Content: '{text}'

Output format as JSON:
[
  {"knowledge": "Principles of forestry resource management", "difficulty": "medium"},
  {"knowledge": "Ecological balance and biodiversity", "difficulty": "complex"}
]
"""

'''
# 角色使命
你是一位专业的文本分析专家，擅长从复杂文本中提取关键知识点并生成可用于模型微调的结构化数据（仅生成问题）。

## 核心任务
根据用户提供的文本（长度：{length}字），生成不少于{q_num}个高质量问题。

## 约束条件（重要！）
✔️ 必须基于文本内容直接生成
✔️ 问题应具有明确答案指向性
✔️ 需覆盖文本的不同方面
❌ 禁止生成假设性、重复或相似问题

## 处理流程
1. 【文本解析】分段处理内容，识别关键实体和核心概念
2. 【问题生成】基于信息密度选择最佳提问点
3. 【质量检查】确保：
   - 问题答案可在原文中找到依据
   - 标签与问题内容强相关
   - 无格式错误
    
## 输出格式
- JSON 数组格式必须正确
- 字段名使用英文双引号
- 输出的 JSON 数组必须严格符合以下结构：
\`\`\`json
[{"id1":{"knowledge":"知识点1","question":"问题1","difficulty":"难度"}},{"id2":{"knowledge":"知识点2","question":"问题2","difficulty":"难度"}}, ...]
\`\`\`

## 输出示例
\`\`\`json
[ "人工智能伦理框架的核心要素":"人工智能伦理框架应包含哪些核心要素？", "民法典对个人数据保护的新规定":"民法典对个人数据保护有哪些？"]
\`\`\`

## 待处理文本
{text}

## 限制
- 必须按照规定的 JSON 格式输出，不要输出任何其他不相关内容
- 生成不少于{q_num}个高质量知识点和问题对
- 问题不要和材料本身相关，例如禁止出现作者、章节、目录等相关问题
- 生成内容须严格属于林业领域，不要生成其它领域内容
'''


# 你是一名林业领域的专家，负责从以下文献中提取与研究背景、关键结论、实验内容相关的核心知识点，并根据支架式教学的理念对知识点进行分层，以支持多层次高质量的试题生成。
KNOWLEDGE_EXTRACTION_ARTICLE_CN = """
你是一名林业领域的专家，需要从以下文献的标题和摘要中提取核心知识点（研究背景、关键结论、重要实验思路或方法等），\
并基于支架式教学的理念对提取内容进行三层次（simple / medium / complex）组织，以支持后续多层次、高质量试题的设计和生成。
请严格按照以下要求提取知识点：
1. **分层原则**：
   - **基础层次（simple）**：提取最基本的概念、术语定义、现象描述。
   - **理解层次（medium）**：提取需要解释、理解或关联的信息，例如机制、因果关系、基本原理等。
   - **分析应用层次（complex）**：提取可进行深度分析或推广应用的内容，如研究思路、关键结论、研究方法的启示。
2. **概括性与逻辑性**：
   - 避免过于零碎的点，优先提取能够体现主要研究脉络或核心内容的知识点。
   - 所提取的知识点之间应具有递进或逻辑关联，反映文章的整体思路。
   - 排除不具备明确主体、缺乏上下文的知识点。
   - **避免生成依赖具体数值、图表或原文数据的知识点**。这类知识点过于具体，无法独立理解和泛化，缺乏教学普适性。
3. **普适性与独立性**：
   - 提取的知识点应具有独立可理解性，无需依赖原文中的数据或数值。
   - 知识点应具有概括性，不得直接引用实验结果或实验数据。

输出严格遵循下列格式，无需生成其它内容：
[
  {{
   "knowledge": "<知识点内容>", 
   "difficulty": "<simple、medium、complex三种标准>"
   }}, ...
]

请基于以下文献内容提取至多 3 个知识点，每个层次最多各一个（可为空）：{text}
"""

KNOWLEDGE_EXTRACTION_WEB_CN = """
你是一位林业领域专家，负责从以下林业相关的文本内容中提取适合构建系统性知识框架的核心知识点。基于支架式教学的概念，以逐步递进的方式提取和组织知识点，并为每个知识点标注其复杂性。

请遵循以下标准进行知识架构的总结和归纳：
1. **层级划分**：按照从简单到复杂的顺序提取知识点，确保知识的分层结构。基础知识点适用于识别和理解层级，高级知识点适用于分析和应用层级。

2. **包含性与关联性**：提取的知识点应具备涵盖性和逻辑性，避免冗余或重复。确保基础知识为上层知识提供支撑，而复杂知识则包含更多细节和应用背景。

输出严格遵循下列格式，无需生成其它内容：
[
  {{
   "knowledge": "<知识点内容>", 
   "difficulty": "<simple、medium、complex三种标准>"
   }}, ...
]
请基于以下文献内容提取知识点并生成对应试题，提取至多 3 个知识点，每个复杂度（简单、中等、复杂）保证最多各一个：{text}
"""

KNOWLEDGE_EXTRACTION_BOOK_CN = """
# 角色使命
你是一位专业的文本分析专家，擅长从复杂文本中提取关键知识点并生成可用于模型微调的结构化数据（仅生成问题）。

## 核心任务
根据用户提供的文本（长度：{length}字），生成不少于{q_num}个高质量问题。

## 约束条件（重要！）
✔️ 必须基于文本内容直接生成
✔️ 问题应具有明确答案指向性
✔️ 需覆盖文本的不同方面
❌ 禁止生成假设性、重复或相似问题

## 处理流程
1. 【文本解析】分段处理内容，识别关键实体和核心概念
2. 【问题生成】基于信息密度选择最佳提问点
3. 【质量检查】确保：
   - 问题答案可在原文中找到依据
   - 标签与问题内容强相关
   - 无格式错误
    
## 输出格式
- JSON 数组格式必须正确
- 字段名使用英文双引号
- 输出的 JSON 数组必须严格符合以下结构：
\`\`\`json
[{"id1":{"knowledge":"知识点1","question":"问题1","difficulty":"难度"}},{"id2":{"knowledge":"知识点2","question":"问题2","difficulty":"难度"}}, ...]
\`\`\`

## 输出示例
\`\`\`json
[ "人工智能伦理框架的核心要素":"人工智能伦理框架应包含哪些核心要素？", "民法典对个人数据保护的新规定":"民法典对个人数据保护有哪些？"]
\`\`\`

## 待处理文本
{text}

## 限制
- 必须按照规定的 JSON 格式输出，不要输出任何其他不相关内容
- 生成不少于{q_num}个高质量知识点和问题对
- 问题不要和材料本身相关，例如禁止出现作者、章节、目录等相关问题
- 生成内容须严格属于林业领域，不要生成其它领域内容
"""



"""
你是一位林业领域专家，负责从以下林业相关的文本内容中提取与林业相关的核心知识点，为后续试题的设计提供支持。
请你基于支架式教学的概念，以逐步递进的方式提取和组织知识点，并为每个知识点标注其复杂性，遵循以下标准：
1. 层级划分：按照从简单到复杂的顺序提取知识点，确保知识的分层结构。基础知识点适用于识别和理解层级，高级知识点适用于分析和应用层级。
2. 包含性与关联性：提取的知识点应具备涵盖性和逻辑性，避免冗余或重复。确保基础知识为上层知识提供支撑，而复杂知识则包含更多细节和应用背景。
3. 只关注林业相关的知识点，避免涉及其他学科或领域的内容。
输出严格遵循下列格式，无需生成其它内容：
[
  {{
   "knowledge": "<知识点内容>", 
   "difficulty": "<simple、medium、complex三种标准>"
   }}, ...
]
请基于以下书籍内容提取知识点并生成对应试题，提取至少3个至多6个知识点，每个复杂度（简单、中等、复杂）保证最少各一个：{text}
"""


# ====== Question Generation Prompts ======
MULTIPLE_CHOICE_PROMPT_EN = """
Based on the following content, create a multiple-choice question that focuses on the specific knowledge point below:
Content: '{text}'
Knowledge Point: '{knowledge}'
Please provide four options and specify the correct answer.
Output format:
"question": "<question here>", "answer": "<answer here>"
"""

SHORT_ANSWER_PROMPT_EN = """
Using the content and specific knowledge point below, create a short-answer question.
Content: '{text}'
Knowledge Point: '{knowledge}'
Provide a detailed answer.
Output format:
"question": "<question here>", "answer": "<answer here>"
"""

OPEN_DISCUSSION_PROMPT_EN = """
Using the content and knowledge point below, create an open-ended discussion question.
Content: '{text}'
Knowledge Point: '{knowledge}'
Provide a detailed answer.
Output format:
"question": "<question here>", "answer": "<answer here>"
"""

# ====== Multiple Choice Prompt ======
MULTIPLE_CHOICE_PROMPT_CN = """
你是一位林业领域教育专家，请你根据所提供的林业相关语料，针对特定的林业相关知识点生成一个包含四个选项的单项选择题，并给出正确答案。
选择题命题准则：
1. **明确性**：题干应表述清晰，避免使用模糊或含糊其辞的语言，问题应具有针对性且直击核心知识点。
2. **科学性**：选项必须基于科学事实，正确答案必须完全符合林业领域的专业知识。
3. **干扰性**：错误选项（干扰项）应具有一定的迷惑性，避免“明显错误”或“离题选项”，同时干扰项应有合理性，但必须与正确答案区分明显。
4. **唯一性**：确保正确答案唯一，且选项内容与题干保持一致，不会引起歧义。
5. **规范性**：使用规范的专业术语，遵守语法和标点规则。

### 示例
原始语料内容：'羊柴是一种主要生长在沙地的固沙植物，具有生物固氮能力、根系发达、耐沙压、抗风蚀等特点。'
知识点：'羊柴的固沙特性'
生成的选择题示例：
'question': '以下关于羊柴固沙植物的特性描述正确的是？\\nA. 无生物固氮能力，根系浅薄，耐沙压，抗风蚀\\nB. 具有生物固氮能力，根系发达，耐沙压，抗风蚀\\nC. 无生物固氮能力，根系浅薄，不耐沙压，不抗风蚀\\nD. 具有生物固氮能力，根系发达，不耐沙压，不抗风蚀',
'answer': 'B'

请根据以下信息生成试题：
原始语料内容：'{text}'
特定知识点：'{knowledge}'

严格遵守输出格式：
'question': '<在此给出问题和选项>', 'answer': '<在此给出答案>'
"""

# ====== Short Answer Prompt ======
SHORT_ANSWER_PROMPT_CN = """\
你是一个林业领域的教育专家，请你根据所提供的林业相关的原始语料内容，针对特定的林业相关知识点生成一个简答题，并提供详细答案。
简答题命题准则：
1. 题目内容应简明扼要，避免过于复杂或包含多重问题。
2. 题目聚焦特定的知识点，确保问题与林业学科相关，并能够有效考察学生对该知识点的理解。
3. 答案应全面、准确，详细阐述关键概念、原理或步骤，并结合实际应用或案例，便于学生理解和记忆。
4. 答案结构清晰，逻辑性强，能够层次分明地展开，避免冗长或模糊的解释。

原始语料内容：'{text}'
知识点：'{knowledge}'

严格遵守输出格式：
'question': '<在此输入问题>', 'answer': '<在此给出答案>'
"""

# ====== Open Discussion Prompt ======
OPEN_DISCUSSION_PROMPT_CN = """
你是一个林业领域的教育专家，请你根据所提供的林业相关的原始语料内容，针对特定的林业相关知识点生成一个开放性论述题，并提供详细答案。
开放性论述题命题准则：
1. 题目聚焦深度思考，问题应涉及林业领域中的重要概念、理论或实践问题，能够激发学生对问题的深入理解与探索。
2. 答案应全面且条理清晰，详细论述相关理论、实践或案例，并结合具体的林业背景，提供有力的证据和分析。
3. 论点逻辑严密，讨论内容应涵盖不同方面并能够展示多维度的视角，且论据充分、分析透彻。
4. 答案应具有一定的启发性，不仅要求知识的复述，还应引导学生思考实际应用和发展方向。

原始语料内容：'{text}'
知识点：'{knowledge}'

严格遵守输出格式：
'question': '<在此输入问题>', 'answer': '<在此给出答案>'
"""


# 统一管理 Prompt 字典
QUESTION_PROMPTS_CN = {
    "knowledge_extraction_article": KNOWLEDGE_EXTRACTION_ARTICLE_CN,
    "knowledge_extraction_web": KNOWLEDGE_EXTRACTION_WEB_CN,
    "knowledge_extraction_book": KNOWLEDGE_EXTRACTION_BOOK_CN,
    "multiple_choice": MULTIPLE_CHOICE_PROMPT_CN,
    "short_answer": SHORT_ANSWER_PROMPT_CN,
    "open_discussion": OPEN_DISCUSSION_PROMPT_CN,
}


'''

# Role: 微调数据集生成专家
## Profile:
- Description: 你是一名微调数据集生成专家，擅长从给定的内容中生成准确的问题答案，确保答案的准确性和相关性。

## Skills   :
1. 答案必须基于给定的内容
2. 答案必须准确，不能胡编乱造
3. 答案必须与问题相关
4. 答案必须符合逻辑
   
## Workflow:
1. Take a deep breath and work on this problem step-by-step.
2. 首先，分析给定的文件内容
3. 然后，从内容中提取关键信息
4. 接着，生成与问题相关的准确答案
5. 最后，确保答案的准确性和相关性

## 参考内容：
${text}

## 问题
${question}

## Constrains:
1. 答案必须基于给定的内容
2. 答案必须准确，必须与问题相关，不能胡编乱造
3. 答案必须充分、详细、包含所有必要的信息、适合微调大模型训练使用

'''