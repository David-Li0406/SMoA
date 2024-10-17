import asyncio
import os
import together
from together import AsyncTogether, Together
import json
import datasets
from functools import partial
import copy
from utils import *
import ast
import random
random.seed(42)
import argparse
import re

os.environ['TOGETHER_API_KEY'] = '28d76a81843ff91a3cb38388b57beee07bf1876001c4ba992a645fc3ed433fb0'
client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

from transformers import AutoTokenizer
from huggingface_hub import login
login("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")


def process_fn(
    item,
    model,
    args,
    reference_models=[],
    temperature=0.7,
    max_tokens=2048,
    tokenizer_dict={},
    rounds=1,
):

    messages = [{"role": "user", "content": item["question"]}]

    references = []
    internal_result = {}
    if reference_models != []:
        token_num_dict = {model_name: 0 if model_name != model else {'debate':0, 'moderate':0, 'aggregate':0} for model_name in reference_models}
    else:
        token_num_dict = {model: 0}

    i_round=0
    if len(references) == 0 and len(reference_models) > 0:

        prev_references = []
        role_prompt_list = []

        for i_round in range(rounds):
            
            if i_round == 0 and args.add_role:
                if not os.path.exists('prompt/{}/role_description.json'.format(args.dataset)):
                    os.makedirs('prompt/{}/'.format(args.dataset), exist_ok=True)
                    messages_role = [{"role": "user", "content": args.role_generation_prompt.format(len(reference_models), len(reference_models), args.task)}]
                    role_prompt, input_messages = generate_together(
                        messages=messages_role,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                    if role_prompt is not None:
                    #     token_num_dict[model]['role'] += len(tokenizer_dict[model].tokenize(role_prompt))
                    #     token_num_dict[model]['role'] += sum([len(tokenizer_dict[model].tokenize(message['content'])) for message in input_messages])
                        role_prompt_list = extract_role_from_output(role_prompt)
                        with open('prompt/{}/role_description.json'.format(args.dataset),'w') as f:
                            json.dump(role_prompt_list,f,indent=2)
                else:
                    role_prompt_list = json.load(open('prompt/{}/role_description.json'.format(args.dataset)))
                    if len(role_prompt_list) < len(reference_models):
                        messages_role = [{"role": "user", "content": args.role_generation_prompt.format(len(reference_models), len(reference_models), args.task)}]
                        role_prompt, input_messages = generate_together(
                            messages=messages_role,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )

                        if role_prompt is not None:
                            role_prompt_list = extract_role_from_output(role_prompt)
                            with open('prompt/{}/role_description.json'.format(args.dataset),'w') as f:
                                json.dump(role_prompt_list,f,indent=2)

            references = []
            internal_result['round_{}'.format(i_round)] = {}
            for idx, role_prompt in enumerate(reference_models):

                reference, input_messages = generate_with_references(
                    model=args.reference_models[idx],
                    messages=messages,
                    system=args.aggreagator_system_prompt,
                    role=role_prompt_list[idx] if args.add_role else '',
                    references=prev_references,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                

                if reference is not None:
                    
                    internal_result['round_{}'.format(i_round)][args.reference_models[idx]] = reference
                    references.append(reference)
                    
                    if args.reference_models[idx] != model:
                        token_num_dict[args.reference_models[idx]] += len(tokenizer_dict[args.reference_models[idx]].tokenize(reference))
                        token_num_dict[args.reference_models[idx]] += sum([len(tokenizer_dict[args.reference_models[idx]].tokenize(message['content'])) for message in input_messages])
                    else:
                        token_num_dict[model]['debate'] += len(tokenizer_dict[model].tokenize(reference))
                        token_num_dict[model]['debate'] += sum([len(tokenizer_dict[model].tokenize(message['content'])) for message in input_messages])

            if references != []:
                if args.moderate_select or args.moderate_end:
                    try:
                        message_moderator = args.moderator_system_prompt.format(args.num_select_response, args.num_select_response, item["question"])
                        for i, reference in enumerate(references):
                            message_moderator += f"Response {i}.\n{reference}"
                        message_moderator += '\nOutput:'
                        message_moderator = [{"role": "user", "content": message_moderator}]

                        judge_output, input_messages = generate_together(
                            messages=message_moderator,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )

                        chosen_responses = [i for i in range(len(references))]
                        end_debate = False

                        token_num_dict[model]['moderate'] += len(tokenizer_dict[model].tokenize(judge_output))
                        token_num_dict[model]['moderate'] += sum([len(tokenizer_dict[model].tokenize(message['content'])) for message in input_messages])

                        if args.moderate_select or args.moderate_end:
                            chosen_responses_output, end_debate_output = extract_indexes_and_indicator_from_output(judge_output)
                            if chosen_responses_output is None or any([chosen_response>=len(references) for chosen_response in chosen_responses_output]):
                                chosen_responses_output = random.sample([i for i in range(len(references))], args.num_select_response)
                            if len(chosen_responses_output) > args.num_select_response:
                                chosen_responses_output = random.sample(chosen_responses_output, args.num_select_response)
                            if args.moderate_select:
                                chosen_responses = chosen_responses_output
                                internal_result['round_{}'.format(i_round)]['chosen response'] = chosen_responses_output
                            if args.moderate_end:
                                end_debate = end_debate_output
                                internal_result['round_{}'.format(i_round)]['end'] = end_debate_output

                        references = [references[i] for i in chosen_responses]
                        if args.moderate_end and end_debate:
                            break
                    except Exception as E:
                        print(E)
                        return {
                            "response": json.dumps({
                                "choices":[{"message":{"content": None, "role":"assistant"}}]
                            }), 
                            "internal_result": internal_result,
                            "generator": model + "-together", 
                            "judge_output": judge_output,
                            "chosen_responses": chosen_responses,
                            "token_num_dict":token_num_dict,
                            "total_round": i_round
                        }

                if i_round < rounds - 1:
                    
                    prev_references = references

                    references = []

    output, input_messages = generate_with_references(
        model=model,
        messages=messages,
        references=references,
        system=args.aggreagator_system_prompt,
    )

    if reference_models != []:
        token_num_dict[model]['aggregate'] += len(tokenizer_dict[model].tokenize(output))
        token_num_dict[model]['aggregate'] += sum([len(tokenizer_dict[model].tokenize(message['content'])) for message in input_messages])
    else:
        token_num_dict[model] += len(tokenizer_dict[model].tokenize(output))
        token_num_dict[model] += sum([len(tokenizer_dict[model].tokenize(message['content'])) for message in input_messages])

    return {
        "response": json.dumps({
            "choices":[{"message":{"content": output, "role":"assistant"}}]
        }), 
        "generator": model + "-together", 
        "judge_output": None, 
        "chosen_responses": None, 
        "token_num_dict":token_num_dict, 
        "total_round": i_round,
        "internal_result": internal_result
    }


def generate_for_ceb(
    args
):
    args.moderator_system_prompt = open('prompt/{}/moderator_system_prompt_v2.txt'.format(args.dataset)).read()
    args.aggreagator_system_prompt=open('prompt/{}/aggreagator_system_prompt.txt'.format(args.dataset)).read()
    args.role_generation_prompt=open('prompt/{}/role_generation_prompt_v2.txt'.format(args.dataset)).read()
    args.task = open('prompt/{}/task.txt'.format(args.dataset)).read()

    if args.reference_models != []:
        tokenizer_dict = {model_name: AutoTokenizer.from_pretrained(get_tokenizer_name(model_name)) for model_name in args.reference_models}
    else:
        tokenizer_dict = {args.model: AutoTokenizer.from_pretrained(get_tokenizer_name(args.model))}

    eval_set = []
    for file in os.listdir('data/{}'.format(args.dataset)):
        data = json.load(open(os.path.join('data/{}'.format(args.dataset), file)))
        for line in data:
            eval_set.append(line)
    
    sample_num = min(args.sample_num, len(eval_set))
    eval_set = random.sample(eval_set, sample_num)

    eval_set = {
        **{key: [item[key] for item in eval_set] for key in eval_set[0].keys()},
        **{'question': [item['prompt'] for item in eval_set]}
    }

    eval_set = datasets.Dataset.from_dict(eval_set)

    eval_set = eval_set.map(
        partial(
            process_fn,
            model=args.model,
            reference_models=args.reference_models,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            rounds=args.rounds,
            tokenizer_dict=tokenizer_dict,
            args=args
        ),
        batched=False,
        num_proc=args.num_proc,
    )

    output_folder = 'output/{}'.format(args.dataset)
    os.makedirs(output_folder, exist_ok=True)

    model_name = args.model.split('/')[1]
    output_path = os.path.join(output_folder, 'case_study_model-{}_reference_model-{}_rounds-{}_num_select_response-{}_add_role-{}_moderate_end-{}_moderate_select-{}_num_models-{}.jsonl'.format(model_name,len(args.reference_models),args.rounds,args.num_select_response,args.add_role,args.moderate_end,args.moderate_select,len(args.reference_models)))

    with open(output_path, "w") as f:
        for item in eval_set:
            f.write(json.dumps(item)+'\n')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Description of your program")
    
    parser.add_argument('--dataset', type=str, help='dataset to use', default='CEB-Conversation-S')
    parser.add_argument('--model', type=str, help='moderater/ aggregator model to use', default='Qwen/Qwen2-72B-Instruct')
    parser.add_argument('--reference_models', type=str, nargs='+', help='debate models to use', default=[])
    parser.add_argument('--rounds', type=int, help='max number of debate round', default=2)
    parser.add_argument('--num_select_response', type=int, help='number of selected response in each iteration', default=2)
    parser.add_argument('--num_proc', type=int, help='max number of process', default=6)
    parser.add_argument('--temperature', type=float, help='temperature', default=0.7)
    parser.add_argument('--max_tokens', type=int, help='max_token', default=2048)
    parser.add_argument('--sample_num', type=int, help='number of samples to use', default=400)


    parser.add_argument('--add_role', action='store_true', help='add different role description to models')
    parser.add_argument('--moderate_end', action='store_true', help='end the debate in advance')
    parser.add_argument('--moderate_select', action='store_true', help='select a sbuset of response for the nextpip install protobuf iteration')

    args = parser.parse_args()
    
    generate_for_ceb(args)

