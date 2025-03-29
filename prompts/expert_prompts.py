# experts.py

# ====== Evaluate Prompts ======
# book
# EVALUATE_QUALITY_BOOK_CN = """
# 你是一位林业领域的命题专家，请根据以下要求评估提供的知识点及其配套试题的质量。确保试题与林业专业高度相关，且知识点与试题之间保持一致性。

# 你需要遵循以下评估标准进行评估：
# 1. Quality Score（1-10分）：整体质量评估
# 	0-2分：严重错误或无关内容，无法考察知识点。
# 	3-5分：部分准确，但存在模糊、错误或缺乏深度的问题。
# 	6-8分：总体准确，表述清晰，但可能存在优化空间。
# 	9-10分：高度准确、清晰且完整，完全符合命题要求。
# 2. Relevance Score（1-10分）：试题与林业领域的相关性
# 	0-2分：与林业领域无关或误导性强。
# 	3-5分：相关性较弱，可能偏离林业领域的核心主题。
# 	6-8分：基本符合林业领域话题，但可能略有泛化。
# 	9-10分：高度相关，与林业核心知识紧密相连。
# 3.Consistency Score（1-10分）：试题与语料原文、知识点的一致性
# 	0-2分：完全偏离知识点，无法正确考察目标内容。
# 	3-5分：部分涉及知识点，但存在较大偏差或考察不充分。
# 	6-8分：较好地匹配知识点，但仍有优化空间。
# 	9-10分：精准匹配知识点，考察全面无偏差。
# 4. Improvement Suggestions: 陈述简短且合理的建议即可，无需评分

# 在评估时请注意：
# 1.按照评估标准进行打分，确保试题符合林业专业知识。
# 2.关注试题类型（Multiple Choice, Short Answer, Open Discussion），并依据不同题型的特点进行针对性评估。不同题型的评分标准应有一定区别：
# Multiple Choice：评估问题及选项是否准确、清晰，且能有效测试知识点。
# Short Answer：评估问题是否能够有效考察知识点的核心，答案是否全面且科学。
# Open Discussion：评估讨论话题的深度，是否能全面覆盖知识点，并进行深入分析。

# 考察知识点：'{knowledge}'
# 对应试题内容：'{response}'
# 试题类型：'{question_type}'

# 无需给出其它回复，请以严格的 JSON 格式输出评估结果，遵循以下要求：
# {{
#   "Quality Score": "<1-10分打分，具体根据评估标准区间给出分数>",
#   "Relevance Score": "<1-10分打分，具体根据评估标准区间给出分数>",
#   "Consistency Score": "<1-10分打分，具体根据评估标准区间给出分数>",
#   "Improvement Suggestions": "<简短且具体的改进建议>"
# }}
# """
EVALUATE_QUALITY_BOOK_CN = """
# 🎯 角色使命
你是一位林业领域的资深命题与评估专家，擅长从专业角度评估各类教学试题质量，并提供优化建议。

## 🧩 核心任务
请基于提供的“知识点”与“试题内容”，从整体质量、相关性、一致性三个维度为该试题打分，并提出改进建议。输出严格的 JSON 格式结果。

## 🧠 评估评分标准
请参考以下标准为每项评分，分值范围为 1-10 分：

### 1. Quality Score（整体质量）
- **0-2 分**：严重错误或无关内容，结构混乱，无法考察知识点。
- **3-5 分**：部分准确，但存在模糊、表达不清或结构不规范问题。
- **6-8 分**：整体较好，表达清晰，基本符合教学要求。
- **9-10 分**：表达清晰、逻辑严谨，完全符合命题规范，具有良好教学价值。

### 2. Relevance Score（专业相关性）
- **0-2 分**：与林业领域无关或内容误导。
- **3-5 分**：相关性较弱，偏离专业主题。
- **6-8 分**：基本相关，但不够聚焦或不深入。
- **9-10 分**：高度相关，聚焦林业核心内容或关键知识点。

### 3. Consistency Score（知识一致性）
- **0-2 分**：与知识点严重不符，无法测查目标内容。
- **3-5 分**：部分涉及知识点，但有偏差或覆盖不充分。
- **6-8 分**：匹配度较好，但细节或深度仍可提升。
- **9-10 分**：精准对齐知识点，覆盖全面，考查到位。

### 4. Improvement Suggestions
请基于评分简要提出**1 条明确具体的建议**，用于后续优化试题。

## 🛠 输入信息
考察知识点：'{knowledge}'  
试题类型：'{question_type}'  
试题内容（题干+选项或参考答案，多选题为csv格式）：  
{response}

## ✅ 输出格式（JSON，严格使用英文引号）
请严格按照以下格式输出，不要生成除 JSON 外的任何解释说明：

```json
{{
  "Quality Score": <1-10>,
  "Relevance Score": <1-10>,
  "Consistency Score": <1-10>,
  "Improvement Suggestions": "<一句话具体建议>"
}}
❌ 禁止添加任何额外文字、格式说明或非 JSON 内容。
"""

# article
EVALUATE_QUALITY_ARTICLE_CN = EVALUATE_QUALITY_BOOK_CN
# web
EVALUATE_QUALITY_WEB_CN = EVALUATE_QUALITY_BOOK_CN

# ========= refine ==========
# book
REFINE_RESPONSE_BOOK_CN = """
你是一位林业领域的命题优化专家。请根据给定的知识点、原始试题和专家改进建议，对试题进行优化重写，使其在专业性、结构清晰度与教学测评效果上更加完善。

## 🛠 输入信息
知识点：'{knowledge}'
原始试题：'{response}'
专家改进建议：'{feedback}'

## ✅ 输出要求
请只输出标准格式，确保改进后试题仍为相同题型（如为选择题则保留四选一结构）。
仅返回如下字段（JSON 字符串格式），无需输出其它内容：

```json
{{
  "question": "<优化后的题干（含选项）或问答题问题，多选题要求csv格式>",
  "answer": "<对应的标准答案>"
}}
❌ 严禁生成附加解释、格式说明、示例等内容。 """

# article
REFINE_RESPONSE_ARTICLE_CN = """
你是一位林业领域的命题优化专家。请依据以下要素，改进当前试题内容，提升其科学性、准确性与教学应用价值。

## 🛠 输入信息
原始试题内容：'{response}'
所考察的知识点：'{knowledge}'
命题专家建议：'{feedback}'

## ✅ 输出格式（JSON）
请仅输出如下格式的标准 JSON，对试题进行优化重写：

```json
{{
  "question": "<改写后的问题或问题+选项，多选题要求csv格式>",
  "answer": "<标准答案>"
}}
❌ 不要输出除 JSON 外的任何自然语言说明。 """


REFINE_RESPONSE_WEB_CN = """
你是一位林业领域的专业命题改进员。请根据提供的林业相关语料、知识点与命题专家建议，对当前试题进行内容优化。

## 🛠 输入信息
原始林业语料：'{text}'
原始试题：'{response}'
知识点：'{knowledge}'
改进建议：'{feedback}'

## ✅ 输出格式（标准 JSON）
```json
{{
  "question": "<优化后的问题与选项或问答题题干，多选题要求csv格式>",
  "answer": "<优化后的标准答案>"
}}
❌ 禁止输出解释说明、格式提示、示例或多余语言。 """

# 统一管理 Prompt 字典
EXPERT_PROMPTS_CN = {
    "evaluate_quality_book": EVALUATE_QUALITY_BOOK_CN,
    "evaluate_quality_article": EVALUATE_QUALITY_ARTICLE_CN,
    "evaluate_quality_web": EVALUATE_QUALITY_WEB_CN,
    "refine_response_book": REFINE_RESPONSE_BOOK_CN,
    "refine_response_article": REFINE_RESPONSE_ARTICLE_CN,
    "refine_response_web": REFINE_RESPONSE_WEB_CN,
}


## EN
EVALUATE_QUALITY_BOOK_EN = """
Evaluate the quality of the following question and answer based on whether it accurately assesses the knowledge point. \
The question is derived from the full content provided. Provide feedback in the specified JSON format:
    Full Text: '{text}'
    Response: '{response}'
Knowledge Point: '{knowledge}'
Output the result in JSON format: 
    {
    'Quality Score': '1-10', 
    'Improvement Suggestions': 'Provide a concise suggestion if needed, leave blank if quality is sufficient'
    }
"""

EVALUATE_QUALITY_ARTICLE_EN = """
For article content, evaluate if the question and answer effectively test the knowledge point. 
Offer suggestions for improvements if needed, considering the broader content context:
Full Text: '{text}'
Response: '{response}'
Knowledge Point: '{knowledge}'
Output the result in JSON format: {'Quality Score': '1-10', 'Improvement Suggestions': 'Provide a concise suggestion if needed; leave blank if quality is sufficient'}
"""

EVALUATE_QUALITY_WEB_EN = """You are a helpful forest assistant. 
For content derived from a forestry-related web source (e.g., Wikipedia, Baidu Baike), \
    assess if the question and answer adequately cover the knowledge point. Suggest refinements if necessary:
Full Text: '{text}'
Response: '{response}'
Knowledge Point: '{knowledge}'
Output the result in JSON format: {'Quality Score': '1-10', 'Improvement Suggestions': 'Provide a concise suggestion if needed; leave blank if quality is sufficient'}
"""


REFINE_RESPONSE_BOOK_EN = """Refine the following question and answer to ensure they thoroughly assess the knowledge point in a book-based context, with reference to the full content. Incorporate the improvement suggestions to enhance quality:
Full Text: '{text}'
Response: '{response}'
Knowledge Point: '{knowledge}'
Feedback: '{feedback}'
Output in JSON format as specified above for the refined question and answer."""

REFINE_RESPONSE_ARTICLE_EN = """
Refine the question and answer for clarity and precision, \
ensuring they test the knowledge point effectively in an article context. 
Consider the broader context and apply the improvement suggestions:
Full Text: '{text}'
Response: '{response}'
Knowledge Point: '{knowledge}'
Feedback: '{feedback}'
Output in JSON format as specified above for the refined question and answer."""

REFINE_RESPONSE_WEB_EN = """For forestry-related web content, enhance the \
    question and answer to better align with the knowledge point, with reference to the full text.
    Incorporate the provided feedback to improve quality:
Full Text: '{text}'
Response: '{response}'
Knowledge Point: '{knowledge}'
Feedback: '{feedback}'
Output in JSON format as specified above for the refined question and answer."""


# EVALUATE_QUALITY_BOOK_CN = """
# 你是一位林业领域的命题专家。请评估以下知识点及其配套试题的质量，确保试题与林业领域高度相关，并且知识点与试题之间保持一致性。
# 你需要遵循以下命题原则进行评分：
#     1.准确性：题目内容和答案必须科学准确，符合林业专业知识。
#     2.清晰性：语言表达清晰明了，避免歧义和模糊不清的表述。
#     3.相关性：题目应紧扣提供的语料和所要考察的知识点。
#     4.完整性：试题应完整全面，答案能够充分解答题目要求。如果试题质量较低，请提出改进建议。

# 参考语料原文：'{text}'
# 考察知识点：'{knowledge}'
# 对应试题内容：'{response}'

# 请按以下格式输出结果：
# 'Quality Score': '1-10分打分', \
# 'Relevance Score': '1-10分打分（知识点以及试题与林业的相关程度）', \
# 'Consistency Score': '1-10分打分（知识点与试题的一致性）', \
# 'Improvement Suggestions': '简短且合理的建议即可'
# """
