REFERENCE_MODELS="Qwen/Qwen2-72B-Instruct Qwen/Qwen1.5-72B-Chat mistralai/Mixtral-8x22B-Instruct-v0.1 databricks/dbrx-instruct"
MODEL=Qwen/Qwen2-72B-Instruct

for DATASET in CEB-Conversation-S CEB-Conversation-T 
    do
        python run_ceb.py --model $MODEL --add_role --moderate_end --moderate_select --reference_models $REFERENCE_MODELS --dataset $DATASET
        python run_ceb.py --model $MODEL --reference_models $REFERENCE_MODELS --dataset $DATASET
        python run_ceb.py --dataset $DATASET
    done