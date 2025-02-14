import os
import importlib
import inspect

# 动态导入所有 *_prompts.py 模块
current_dir = os.path.dirname(__file__)

for file in os.listdir(current_dir):
    if file.endswith("_prompts.py") and file != "__init__.py":
        module_name = f".{file[:-3]}"
        module = importlib.import_module(module_name, __package__)

        # 获取所有以 'CN' 结尾的变量
        for name, value in inspect.getmembers(module):
            if name.endswith("_CN"):
                globals()[name] = value

# 定义 __all__ 避免不必要的变量暴露
__all__ = [name for name in globals() if name.endswith("_CN")]


# from expert_prompts import (
#     EVALUATE_QUALITY_BOOK_CN,
#     EVALUATE_QUALITY_ARTICLE_CN,
#     EVALUATE_QUALITY_WEB_CN,
#     REFINE_RESPONSE_BOOK_CN,
#     REFINE_RESPONSE_ARTICLE_CN,
#     REFINE_RESPONSE_WEB_CN
# )

# from finl_eval_prompt import GRADE_PROMPT_CN

# from question_prompt import (
#     KNOWLEDGE_EXTRACTION_ARTICLE_CN,
#     KNOWLEDGE_EXTRACTION_WEB_CN,
#     KNOWLEDGE_EXTRACTION_BOOK_CN,

#     MULTIPLE_CHOICE_PROMPT_CN,
#     SHORT_ANSWER_PROMPT_CN,
#     OPEN_DISCUSSION_PROMPT_CN
