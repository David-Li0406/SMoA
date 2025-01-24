import json
import math
import argparse

def split(args):

    # output1='/home/dawei/projects/dawei/MoA-Debate/output/math_understand/model-Qwen2-72B-Instruct_reference_model-0_rounds-2_num_select_response-2_add_role-False_moderate_end-False_moderate_select-False_num_models-0.jsonl'
    # output2='/home/dawei/projects/dawei/MoA-Debate/output/math_understand/model-Qwen2-72B-Instruct_reference_model-4_rounds-2_num_select_response-2_add_role-False_moderate_end-False_moderate_select-False_num_models-4.jsonl'
    # output3='/home/dawei/projects/dawei/MoA-Debate/output/math_understand/model-Qwen2-72B-Instruct_reference_model-4_rounds-2_num_select_response-2_add_role-True_moderate_end-True_moderate_select-True_num_models-4.jsonl'

    chunk=5

    # for output in [output1,output2,output3]:
    eval_set = []
    with open(args.path) as f:
        for i, line in enumerate(f.readlines()):
            eval_set.append(line)
        eval_set_new = []
        for i in range(math.floor(len(eval_set)/chunk)):
            eval_set_new.append(eval_set[i*chunk:(i+1)*chunk])
        eval_set_new.append(eval_set[(i+1)*chunk:])
        
        for i, eval_set_sub in enumerate(eval_set_new):
            with open(args.path+".{}".format(i), 'w') as f:
                for item in eval_set_sub:
                    f.write(item)


def merge(args):
    # output1='/home/dawei/projects/dawei/MoA-Debate/metrics/math_understand/model-Qwen2-72B-Instruct_reference_model-0_rounds-2_num_select_response-2_add_role-False_moderate_end-False_moderate_select-False_num_models-0.jsonl'
    # output2='/home/dawei/projects/dawei/MoA-Debate/metrics/math_understand/model-Qwen2-72B-Instruct_reference_model-4_rounds-2_num_select_response-2_add_role-False_moderate_end-False_moderate_select-False_num_models-4.jsonl'
    # output3='/home/dawei/projects/dawei/MoA-Debate/metrics/math_understand/model-Qwen2-72B-Instruct_reference_model-4_rounds-2_num_select_response-2_add_role-True_moderate_end-True_moderate_select-True_num_models-4.jsonl'
    # outputs=[output1,output2,output3]
    # models = ['baseline', 'moa', 'ours']

    # for output, model in zip(outputs, models):
    total_correct = 0
    total_valid = 0
    for i in range(23):
        metric = json.load(open(args.path+".{}".format(i)))
        total_correct += metric['accuracy'] * metric['total_valid_answers']
        total_valid += metric['total_valid_answers']

    accuracy_valid = total_correct/total_valid
    accuracy_all = total_correct/200
    with open(args.output, 'w') as f:
        json.dump({
            'accuracy_valid':accuracy_valid,
            'accuracy_all':accuracy_all
        },f)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument('--path', type=str, help='path to read', default='')
    parser.add_argument('--output', type=str, help='output path to use', default='')
    parser.add_argument('--mode', type=str, help='mode to use', default='split')
    args = parser.parse_args()
    if args.mode == 'split':
        split(args)
    else:
        merge(args)