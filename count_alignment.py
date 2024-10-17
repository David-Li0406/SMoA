import argparse
import json
import os


def count_for_mmlu(args):
    if args.output_path != '':
        output_path = 'output/{}'.format(args.output_path)
    else:
        output_folder = 'output/{}'.format(args.dataset)
        model_name = args.model.split('/')[1]
        output_path = os.path.join(output_folder, 'model-{}_reference_model-{}_rounds-{}_num_select_response-{}_add_role-{}_moderate_end-{}_moderate_select-{}_num_models-{}.jsonl'.format(model_name,len(args.reference_models),args.rounds,args.num_select_response,args.add_role,args.moderate_end,args.moderate_select,len(args.reference_models)))

    eval_set = json.load(open(output_path))
    
    total = 0
    for item in eval_set:
        for k,v in item['token_num_dict'].items():
            if isinstance(v,int):
                total += v
            else:
                for kk, vv in v.items():
                    total += vv
    

    if args.output_path != '':
        write_path = 'count/{}'.format(args.output_path)
    else:
        write_folder = 'count/{}'.format(args.dataset)
        os.makedirs(write_folder, exist_ok=True)
        output_path = 'model-{}_reference_model-{}_rounds-{}_num_select_response-{}_add_role-{}_moderate_end-{}_moderate_select-{}_num_models-{}.jsonl'.format(model_name,len(args.reference_models),args.rounds,args.num_select_response,args.add_role,args.moderate_end,args.moderate_select,len(args.reference_models))
        write_path = os.path.join(write_folder, output_path)
    with open(write_path, 'w') as f:
        json.dump({'total_token': total}, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Description of your program")
    
    parser.add_argument('--dataset', type=str, help='dataset to use', default='alignment')
    parser.add_argument('--model', type=str, help='moderater/ aggregator model to use', default='Qwen/Qwen2-72B-Instruct')
    parser.add_argument('--reference_models', type=str, nargs='+', help='debate models to use', default=[])
    parser.add_argument('--rounds', type=int, help='max number of debate round', default=2)
    parser.add_argument('--num_select_response', type=int, help='number of selected response in each iteration', default=2)
    parser.add_argument('--num_proc', type=int, help='max number of process', default=6)
    parser.add_argument('--temperature', type=float, help='temperature', default=0.7)
    parser.add_argument('--max_tokens', type=int, help='max_token', default=2048)
    parser.add_argument('--sample_num', type=int, help='number of samples to use', default=200)
    parser.add_argument('--output_path', type=str, help='output path to use', default='')


    parser.add_argument('--add_role', action='store_true', help='add different role description to models')
    parser.add_argument('--moderate_end', action='store_true', help='end the debate in advance')
    parser.add_argument('--moderate_select', action='store_true', help='select a sbuset of response for the nextpip install protobuf iteration')

    args = parser.parse_args()
    
    count_for_mmlu(args)