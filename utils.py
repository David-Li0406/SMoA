import os
import json
import time
import requests
import openai
import copy
import re
import ast
from openai.types.chat import ChatCompletion

from loguru import logger


DEBUG = int(os.environ.get("DEBUG", "0"))


def generate_together(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
    streaming=False,
    n=1,
):

    output = None

    for sleep_time in [1, 2, 4, 8, 16, 32]:

        try:

            endpoint = "https://api.together.xyz/v1/chat/completions"

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}...`) to `{model}`."
                )

            res = requests.post(
                endpoint,
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": (temperature if temperature > 1e-4 else 0),
                    "messages": messages,
                    "n": n,
                },
                headers={
                    "Authorization": f"Bearer {os.environ.get('TOGETHER_API_KEY')}",
                },
            )
            if "error" in res.json():
                logger.error(res.json())
                if res.json()["error"]["type"] == "invalid_request_error":
                    logger.info("Input + output is longer than max_position_id.")
                    return None, None

            if n == 1:
                output = res.json()["choices"][0]["message"]["content"].strip()
            else:
                output = [item["message"]["content"].strip() for item in res.json()["choices"]]

            break

        except Exception as e:
            logger.error(e)
            if DEBUG:
                logger.debug(f"Msgs: `{messages}`")

            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:

        return None,None

    if DEBUG:
        logger.debug(f"Output: `{output[:20]}...`.")

    return output, messages


def generate_together_stream(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):
    endpoint = "https://api.together.xyz/v1"
    client = openai.OpenAI(
        api_key=os.environ.get("TOGETHER_API_KEY"), base_url=endpoint
    )
    endpoint = "https://api.together.xyz/v1/chat/completions"
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature if temperature > 1e-4 else 0,
        max_tokens=max_tokens,
        stream=True,  # this time, we set stream=True
    )

    return response


def generate_openai(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):

    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = completion.choices[0].message.content
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    output = output.strip()

    return output


def generate_gitaigc(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):
    api_key = 'sk-jfhonAkNxKzfViMm5dD93d8d8a0844D7B4160bE837A024Fa'
    url = 'https://gitaigc.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }

    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:
            
            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            # completion = client.chat.completions.create(
            #     model=model,
            #     messages=messages,
            #     temperature=temperature,
            #     max_tokens=max_tokens,
            # )
            completion = requests.post(
                url, headers=headers, json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            )
            completion = ChatCompletion(**(completion.json()))

            output = completion.choices[0].message.content
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    output = output.strip()

    return output, messages


def generate_with_references(
    model,
    messages,
    system,
    role='',
    references=[],
    max_tokens=2048,
    temperature=0.7,
    generate_fn=generate_together,
):

    if len(references) > 0:
        messages = inject_references_to_messages(messages, references, system)

    if role != '':
        messages = inject_role_to_messages(messages, role)


    return generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def extract_indexes_and_indicator_from_output(output):
    # Regular expressions to extract "chosen responses" and "end debate"
    chosen_responses_pattern = re.compile(r'"chosen responses": (\[.*?\])')
    end_debate_pattern = re.compile(r'"end debate": (True|False|true|false)')

    # Extract the chosen responses
    chosen_responses_match = chosen_responses_pattern.search(output)
    chosen_responses = ast.literal_eval(chosen_responses_match.group(1)) if chosen_responses_match else None

    # Extract the end debate value
    end_debate_match = end_debate_pattern.search(output)
    end_debate = True if end_debate_match and end_debate_match.group(1).lower() == 'true' else False

    return chosen_responses, end_debate



def extract_role_from_output(output):
    roles = []
    new_role = ""
    for line in output.strip().split('\n'):
        if "Generated Role Description" in line:
            if new_role != "":
                roles.append(new_role.strip())
            new_role = ""
            continue
        new_role += line
    roles.append(new_role.strip())

    return roles


def inject_role_to_messages(
    messages,
    role,
):

    messages = [{"role": "system", "content": role}] + messages
    return messages


def inject_references_to_messages(
    messages,
    references,
    system,
):

    messages = copy.deepcopy(messages)

    for i, reference in enumerate(references):

        system += f"Response {i}\n. {reference}"

    if messages[0]["role"] == "system":

        messages[0]["content"] += "\n\n" + system

    else:

        messages = [{"role": "system", "content": system}] + messages

    return messages

def get_tokenizer_name(model_name):
    if model_name == "meta-llama/Llama-3-70b-chat-hf":
        return "meta-llama/Meta-Llama-3-70B-Instruct"
    elif model_name == "microsoft/WizardLM-2-8x22B":
        return "alpindale/WizardLM-2-8x22B"
    else:
        return model_name