import openai
import numpy as np
import json
import time
import sys
import os
from openai import OpenAI
import requests

try:
    import google.generativeai as genai
    from anthropic import Anthropic
except:
    pass


def get_openai_embedding(texts, model="text-embedding-ada-002"):
    texts = [text.replace("\n", " ") for text in texts]
    return np.array(
        [
            openai.Embedding.create(input=texts, model=model)["data"][i]["embedding"]
            for i in range(len(texts))
        ]
    )


def set_anthropic_key():
    pass


def set_gemini_key():

    # Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


def set_openai_key():
    openai.api_key = os.environ["OPENAI_API_KEY"]


def run_json_trials(
    query,
    num_gen=1,
    num_tokens_request=1000,
    model="davinci",
    use_16k=False,
    temperature=1.0,
    wait_time=1,
    examples=None,
    input=None,
):

    run_loop = True
    counter = 0
    while run_loop:
        try:
            if examples is not None and input is not None:
                output = run_chatgpt_with_examples(
                    query,
                    examples,
                    input,
                    num_gen=num_gen,
                    wait_time=wait_time,
                    num_tokens_request=num_tokens_request,
                    use_16k=use_16k,
                    temperature=temperature,
                ).strip()
            else:
                output = run_chatgpt(
                    query,
                    num_gen=num_gen,
                    wait_time=wait_time,
                    model=model,
                    num_tokens_request=num_tokens_request,
                    use_16k=use_16k,
                    temperature=temperature,
                )
            output = output.replace("json", "")  # this frequently happens
            facts = json.loads(output.strip())
            run_loop = False
        except json.decoder.JSONDecodeError:
            counter += 1
            time.sleep(1)
            print("Retrying to avoid JsonDecodeError, trial %s ..." % counter)
            print(output)
            if counter == 10:
                print("Exiting after 10 trials")
                sys.exit()
            continue
    return facts


def run_claude(query, max_new_tokens, model_name):

    if model_name == "claude-sonnet":
        model_name = "claude-3-sonnet-20240229"
    elif model_name == "claude-haiku":
        model_name = "claude-3-haiku-20240307"

    client = Anthropic(
        # This is the default and can be omitted
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    # print(query)
    message = client.messages.create(
        max_tokens=max_new_tokens,
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        model=model_name,
    )
    print(message.content)
    return message.content[0].text


def run_gemini(model, content: str, max_tokens: int = 0):

    try:
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
        return None


def run_chatgpt_with_examples(
    query,
    examples,
    input,
    num_gen=1,
    num_tokens_request=1000,
    use_16k=False,
    wait_time=1,
    temperature=1.0,
):

    completion = None

    messages = [{"role": "system", "content": query}]
    for inp, out in examples:
        messages.append({"role": "user", "content": inp})
        messages.append({"role": "system", "content": out})
    messages.append({"role": "user", "content": input})

    while completion is None:
        wait_time = wait_time * 2
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo" if not use_16k else "gpt-3.5-turbo-16k",
                temperature=temperature,
                max_tokens=num_tokens_request,
                n=num_gen,
                messages=messages,
            )
        except openai.error.APIError as e:
            # Handle API error here, e.g. retry or log
            print(
                f"OpenAI API returned an API Error: {e}; waiting for {wait_time} seconds"
            )
            time.sleep(wait_time)
            pass
        except openai.error.APIConnectionError as e:
            # Handle connection error here
            print(
                f"Failed to connect to OpenAI API: {e}; waiting for {wait_time} seconds"
            )
            time.sleep(wait_time)
            pass
        except openai.error.RateLimitError as e:
            # Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass
        except openai.error.ServiceUnavailableError as e:
            # Handle rate limit error (we recommend using exponential backoff)
            print(
                f"OpenAI API request exceeded rate limit: {e}; waiting for {wait_time} seconds"
            )
            time.sleep(wait_time)
            pass

    return completion.choices[0].message.content


def run_chatgpt(
    query,
    num_gen=1,
    num_tokens_request=1000,
    model="chatgpt",
    use_16k=False,
    temperature=1.0,
    wait_time=1,
):
    """
    通用的 ChatGPT 和 OpenAI API 调用函数，支持多种模型。
    """

    client = OpenAI(
        api_key="sk-I2PKH5ezPg9zEk6ny8T0HQtkm4g24ALXd6akjRtcuHHfkfrb",
        base_url="https://xiaoai.plus/v1",
        # "sk-Fbqjug5suyrFtiNxQ87Up4Wx02NLP2bRVhc2orBA8lEHsqot"
    )  # This is the default and can be omitted

    completion = None
    while completion is None:

        if "gpt-3.5-turbo" in model:
            messages = [{"role": "system", "content": query}]
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo-16k" if use_16k else "gpt-3.5-turbo",
                temperature=temperature,
                max_tokens=num_tokens_request,
                n=num_gen,
                messages=messages,
            )
        elif "gpt-4" in model:
            messages = [{"role": "user", "content": query}]
            completion = client.chat.completions.create(
                model="gpt-4-1106-preview",
                temperature=temperature,
                max_tokens=num_tokens_request,
                n=num_gen,
                messages=messages,
            )
        elif model == "chatgpt_o1-preview":
            messages = [{"role": "user", "content": query}]
            completion = client.chat.completions.create(
                model="o1-preview",
                temperature=temperature,
                max_tokens=num_tokens_request,
                n=num_gen,
                messages=messages,
            )
        elif model == "davinci":
            completion = openai.Completion.create(
                model="text-davinci-003",
                prompt=query,
                temperature=temperature,
                max_tokens=num_tokens_request,
                n=num_gen,
            )
        else:
            raise ValueError(f"Model {model} is not supported.")

    return completion.choices[0].message.content


# qwen2
def run_qwen(query, num_gen=1, num_tokens_request=1000, wait_time=1, temperature=0.8):

    client = OpenAI(
        api_key="sk-9fca3e0e00994b96835cf550bb254ba0",  # 使用您的 Dashscope API 密钥
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-plus",  # qwen-max0.02 0.06 qwen-plus0.0008 0.002 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models  qwen-max-0919  qwen-max  qwen2.5-72b-instruct
        temperature=temperature,
        # max_tokens = num_tokens_request,
        n=num_gen,
        messages=[{"role": "user", "content": query}],
    )

    return completion.choices[0].message.content


# deepseek
def run_ds(query, model="deepseek-r1", temperature=0.7, num_gen=1):
    """
    调用 DeepSeek API 进行对话。

    参数:
        query (str): 用户输入的文本
        model (str): 使用的 DeepSeek 模型 (默认: deepseek-r1)
        temperature (float): 采样温度，控制生成的多样性 (默认: 0.7)
        num_gen (int): 生成的回答数量 (默认: 1)

    返回:
        tuple: (思考过程, 最终答案)
    """

    client = OpenAI(
        api_key="sk-9fca3e0e00994b96835cf550bb254ba0", 
        # api_key=os.getenv("sk-9fca3e0e00994b96835cf550bb254ba0"),  # API Key 读取方式
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="deepseek-r1",
        temperature=temperature,
        n=num_gen,
        messages=[{"role": "user", "content": query}],
    )

    # 提取思考过程和最终答案
    reasoning_content = completion.choices[0].message.reasoning_content
    final_content = completion.choices[0].message.content

    # 组织返回结果
    result = {
        "reasoning": completion.choices[0].message.reasoning_content,  # 思考过程
        "answer": completion.choices[0].message.content  # 最终答案
    }

    return result
    

def run_agent(prompt, model="qwen", num_gen=1, temperature=1):
    """调用大模型进行生成"""

    if "qwen" in model:
        response = run_qwen(prompt, num_gen=num_gen, temperature=temperature)
    elif "gpt" in model:
        response = run_chatgpt(
            prompt, model=model, num_gen=num_gen, temperature=temperature
        )
    elif "deepseek" in model:
        response = run_ds(prompt)
    else:
        raise ValueError(f"Unsupported model: {model}")

    return response

# babel -ipdbgt /home/zdx/xxx.pdbgt -opdb /home/zdx/xxx.pdb