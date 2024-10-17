REFERENCE_MODELS="Qwen/Qwen2-72B-Instruct Qwen/Qwen1.5-72B-Chat mistralai/Mixtral-8x22B-Instruct-v0.1 databricks/dbrx-instruct"

for MODEL in Qwen/Qwen1.5-72B-Chat mistralai/Mixtral-8x22B-Instruct-v0.1 databricks/dbrx-instruct
    do
        for DATASET in CEB-Conversation-S CEB-Conversation-T code_contests_regular tool_use_plan_triple_with_reasoning math_understand
            do
                python count_mmau.py --model $MODEL --add_role --moderate_end --moderate_select --reference_models $REFERENCE_MODELS --dataset $DATASET
                python count_mmau.py --model $MODEL --reference_models $REFERENCE_MODELS --dataset $DATASET
                python count_mmau.py --model $MODEL --dataset $DATASET
            done

        for DATASET in alignment
            do
                python count_alignment.py --model $MODEL --add_role --moderate_end --moderate_select --reference_models $REFERENCE_MODELS --dataset $DATASET
                python count_alignment.py --model $MODEL --reference_models $REFERENCE_MODELS --dataset $DATASET
                python count_alignment.py --model $MODEL --dataset $DATASET
            done
    
    done