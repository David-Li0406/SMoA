export CODE_KAGGLE_DATA_DIR=code_kaggle_data_dir
REFERENCE_MODELS="Qwen/Qwen2-72B-Instruct Qwen/Qwen1.5-72B-Chat mistralai/Mixtral-8x22B-Instruct-v0.1 databricks/dbrx-instruct"

for MODEL in Qwen/Qwen2-72B-Instruct Qwen/Qwen1.5-72B-Chat mistralai/Mixtral-8x22B-Instruct-v0.1 databricks/dbrx-instruct
    do
        for DATASET in code_contests_regular tool_use_plan_triple_with_reasoning math_understand
            do
                python run_mmau.py --model $MODEL --add_role --moderate_end --moderate_select --reference_models $REFERENCE_MODELS --dataset $DATASET
                python run_mmau.py --model $MODEL --reference_models $REFERENCE_MODELS --dataset $DATASET
                python run_mmau.py --dataset $DATASET --model $MODEL
            done

        for DATASET in CEB-Conversation-S CEB-Conversation-T 
            do
                python run_ceb.py --model $MODEL --add_role --moderate_end --moderate_select --reference_models $REFERENCE_MODELS --dataset $DATASET
                python run_ceb.py --model $MODEL --reference_models $REFERENCE_MODELS --dataset $DATASET
                python run_ceb.py --dataset $DATASET --model $MODEL
            done

        for DATASET in alignment 
            do
                python run_alignment.py --model $MODEL --add_role --moderate_end --moderate_select --reference_models $REFERENCE_MODELS --dataset $DATASET
                python run_alignment.py --model $MODEL --reference_models $REFERENCE_MODELS --dataset $DATASET
                python run_alignment.py --dataset $DATASET --model $MODEL
            done
    done