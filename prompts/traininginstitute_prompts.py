CONVERT_TO_CONVERSATION_BOOK_EN = """
Assume you are a forestry expert. 
Based on the background information below, convert the test question and answer \
    into a structured response that directly addresses the question. Avoid exam-style \
    language (such as 'correct answer') and respond in a concise, professional tone that \
    clearly explains the knowledge point. Use the background information to add depth to the response where relevant. 

Background: '{text}'
Input: '{response}'

, Output in the format:
"input": "<question here>", "output": "<answer here>"
"""
# CONVERT_TO_CONVERSATION_WEB_EN

# =========== 口语化 ============
# book
CONVERT_TO_CONVERSATION_BOOK_CN = """
你是一位经验丰富的林业专家。
请基于以下提供的信息，将给定的**试题内容**转换为结构化的日常对话形式。转换时请遵循以下要求：
    1. **自然对话风格**：将试题改写为自然的、指令风格的数据格式，避免使用任何考试或测试的语言（如“选择正确的答案”或“判断对错”）。
    2. **丰富回答内容**：在回答中适当引用和融合背景信息，但不要直接复制粘贴。确保回答专业、准确，同时易于理解。
    3. **避免非正式语言**：使用正式且礼貌的语言，不要使用俚语或过于口语化的表达。
    4. **结构化输出**：按照指定的格式输出结果，确保格式正确。

原始输入试题：'{response}'

输出格式如下：
{{
    "input": "<问题在此>", 
    "output": "<回答在此>"
}}

例如：
原始输入试题：下列哪种树种更适合在弱光环境下生长？A. 松树 B. 云杉 C. 樟树 D. 榆树 （正确答案：B. 云杉）
输出:
{{
    "input": "你知道哪种树更适合在弱光环境下很好地生长吗？", 
    "output": "云杉是一种非常适合弱光环境的树种。它们通常生长在密集的森林中，能够在有限的阳光条件下进行光合作用，保持健康的生长状态。"
}}

"""
CONVERT_TO_CONVERSATION_ARTICLE_CN = CONVERT_TO_CONVERSATION_BOOK_CN
CONVERT_TO_CONVERSATION_WEB_CN = CONVERT_TO_CONVERSATION_BOOK_CN

# ===============  CoT =============
# book
GENERATE_CHAIN_OF_THOUGHT_PROMPT_BOOK_PROMPT_EN = """
You are a virtual teacher with expertise in forestry. 
Based on the question and its answer provided below, construct a detailed chain of thought to guide a student through the reasoning process to arrive at the correct answer. 
Focus on clarity, logical steps, and providing necessary background context.
Question: '{text}'
Answer: '{response}'
Output the chain of thought as:
"chain_of_thought": "<step-by-step reasoning here>"
"""

GENERATE_CHAIN_OF_THOUGHT_BOOK_CN = """
你是一位专注于林业领域的虚拟教师，擅长引导学生通过清晰的思维链进行深入学习。\
请基于以下提供的问题和答案，构建一个逻辑严谨、层次分明的思维链，逐步引导学生推理并得出正确结论。\
每个步骤都应有理有据，并结合必要的背景信息，帮助学生更全面地理解和掌握知识点，确保推理过程清晰易懂且循序渐进。

思维链构建原则：
    1.逻辑性：思维链的每个步骤都要有明确的逻辑关系，不能跳跃或含糊。从已知信息逐步推导到答案，确保每个环节都有依据。
    2.分层次：将问题分解为多个可理解的小步骤，逐层递进。从基础概念到核心原理，逐步引导学生理解答案。
    3.背景支持：提供必要的背景信息，帮助学生建立相关知识连接。解释每个步骤背后的原理或原因，确保学生能理解而非死记硬背。
    4.启发性：每个推理步骤应引导学生进行思考，而非直接给出结论。适时提出问题，引发学生自主思考和分析。
    5.明确结论：在思维链的最后一步，清晰地总结并指出答案。确保结论与之前的推理过程保持一致，避免歧义。

试题及答案：'{response}'

请按以下格式输出思维链：
"CoT": "<在此写出逐步推理过程>"

示例： 
试题及答案：'question': '在弱光环境下，哪种树种更适合生长？', 'answer': '云杉'
"CoT": "首先，不同树种在光照需求上存在差异。一些树种需要充足的阳光才能正常生长，而另一些树种则能够适应较弱的光照环境。\n\n其次，弱光环境通常存在于森林较为密集的区域，这些区域阳光难以透过茂密的树冠直达地面。\n\n再者，云杉是一种典型的耐荫树种，它具有较强的弱光适应能力，能够在阳光较少的环境下进行有效的光合作用。\n\n因此，通过对比不同树种的光照需求和生存特性，可以得出结论：云杉更适合在弱光环境下生长。"

"""

GENERATE_CHAIN_OF_THOUGHT_ARTICLE_CN = GENERATE_CHAIN_OF_THOUGHT_BOOK_CN
GENERATE_CHAIN_OF_THOUGHT_WEB_CN = GENERATE_CHAIN_OF_THOUGHT_BOOK_CN

# 统一管理 Prompt 字典
CONVERSATION_PROMPTS_CN = {
    "convert_to_conversation_book": CONVERT_TO_CONVERSATION_BOOK_CN,
    "convert_to_conversation_article": CONVERT_TO_CONVERSATION_ARTICLE_CN,
    "convert_to_conversation_web": CONVERT_TO_CONVERSATION_WEB_CN,
    "generate_chain_of_thought_book": GENERATE_CHAIN_OF_THOUGHT_BOOK_CN,
    "generate_chain_of_thought_article": GENERATE_CHAIN_OF_THOUGHT_ARTICLE_CN,
    "generate_chain_of_thought_web": GENERATE_CHAIN_OF_THOUGHT_WEB_CN,
}
