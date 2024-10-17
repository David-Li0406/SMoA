for MODEL in Qwen2-72B-Instruct Qwen1.5-72B-Chat Mixtral-8x22B-Instruct-v0.1 dbrx-instruct

    do

        EVAL_SET_BASELINE=model-${MODEL}_reference_model-0_rounds-2_num_select_response-2_add_role-False_moderate_end-False_moderate_select-False_num_models-0.jsonl
        EVAL_SET_MOA=model-${MODEL}_reference_model-4_rounds-2_num_select_response-2_add_role-False_moderate_end-False_moderate_select-False_num_models-4.jsonl
        EVAL_SET_OUR=model-${MODEL}_reference_model-4_rounds-2_num_select_response-2_add_role-True_moderate_end-True_moderate_select-True_num_models-4.jsonl

        for EVAL_SET in $EVAL_SET_BASELINE $EVAL_SET_MOA $EVAL_SET_OUR
            do
                just_eval \
                    --mode "score_multi" \
                    --gpt_model "gpt-4o-2024-05-13" \
                    --model_output_file output/alignment/${EVAL_SET} \
                    --eval_output_file metrics/alignment/${EVAL_SET}.multi \
                    --start_idx 0 \
                    --end_idx 800

                just_eval --report_only --mode "score_multi" \
                        --eval_output_file metrics/alignment/${EVAL_SET}.multi

                
                just_eval \
                --mode "score_safety" \
                --model "gpt-3.5-turbo-0613" \
                --model_output_file output/alignment/${EVAL_SET} \
                --eval_output_file metrics/alignment/${EVAL_SET}.safety \
                --start_idx 800 \
                --end_idx 1000
        
                just_eval --report_only --mode "score_safety" \
                    --eval_output_file metrics/alignment/${EVAL_SET}.safety

            done
    
done


for MODEL in Qwen2-72B-Instruct Qwen1.5-72B-Chat Mixtral-8x22B-Instruct-v0.1 dbrx-instruct
    do
        for DATASET in CEB-Conversation-S CEB-Conversation-T 
            do  
                python eval_ceb.py --model $MODEL --add_role --moderate_end --moderate_select --reference_models $REFERENCE_MODELS --dataset $DATASET
                python eval_ceb.py --model $MODEL --reference_models $REFERENCE_MODELS --dataset $DATASET
                python eval_ceb.py --dataset $DATASET --model $MODEL

            done
    done

for MODEL in Qwen2-72B-Instruct Qwen1.5-72B-Chat Mixtral-8x22B-Instruct-v0.1 dbrx-instruct
    do
        EVAL_SET_BASELINE=model-${MODEL}_reference_model-0_rounds-2_num_select_response-2_add_role-False_moderate_end-False_moderate_select-False_num_models-0.jsonl
        EVAL_SET_MOA=model-${MODEL}_reference_model-4_rounds-2_num_select_response-2_add_role-False_moderate_end-False_moderate_select-False_num_models-4.jsonl
        EVAL_SET_OUR=model-${MODEL}_reference_model-4_rounds-2_num_select_response-2_add_role-True_moderate_end-True_moderate_select-True_num_models-4.jsonl
        for DATASET in code_contests_regular tool_use_plan_triple_with_reasoning math_understand
            do
                for EVAL_SET in $EVAL_SET_BASELINE $EVAL_SET_MOA $EVAL_SET_OUR
                    do
                        if [ "$DATASET" = "code_contests_regular" ]; then
                            python3 -m axlearn.open_api.evaluator \
                                --input_file ./output/code_contests_regular/$EVAL_SET \
                                --output_file ./metrics/code_contests_regular/$EVAL_SET \
                                --metric_name code_contests \
                                --grader_model gpt-4o-2024-05-13

                        elif [ "$DATASET" = "tool_use_plan_triple_with_reasoning" ]; then
                            mkdir ./metrics/tool_use_plan_triple_with_reasoning
                            python3 -m axlearn.open_api.evaluator \
                                --input_file ./output/tool_use_plan_triple_with_reasoning/$EVAL_SET \
                                --output_file ./metrics/tool_use_plan_triple_with_reasoning/$EVAL_SET \
                                --metric_name tool_use_plan

                        elif [ "$DATASET" = "math_understand" ]; then
                            python process_math.py --mode split --path ./output/math_understand/${EVAL_SET}

                            for i in {0..22}
                                do
                                    python3 -m axlearn.open_api.evaluator \
                                        --input_file ./output/math_understand/${EVAL_SET}.${i} \
                                        --output_file ./metrics/math_understand/${EVAL_SET}.${i} \
                                        --metric_name math \
                                        --grader_model gpt-4o-2024-05-13
                                done

                            mkdir ./metrics/math_understand_all/
                            python process_math.py --mode aggregate --path ./metrics/math_understand/${EVAL_SET} --output ./metrics/math_understand_all/${EVAL_SET}
                    done
            done
    done