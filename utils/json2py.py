import json

with open('/home/wyp/project/ForestLLM/prompts/traininginstitute_prompts.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

with open('/home/wyp/project/ForestLLM/prompts/traininginstitute_prompts.py', 'w', encoding='utf-8') as py_file:
    for key, section in data.items():
        for sub_key, prompt in section.items():
            for lang, text in prompt.items():
                var_name = f"{key.upper()}_{sub_key.upper()}_{lang.upper()}"
                py_file.write(f'{var_name} = """{text}"""\n\n')
