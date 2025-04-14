
# Set number of processes (1 per GPU)
num_process=2
master_port=29500  # or any free port

# Configure Accelerate for Kaggle (might require setup beforehand)
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \ 
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1_small_small.csv \
  --model_id ETTh1_small_small \
  --model TimeLLM \
  --data Traffic \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des Exp \
  --itr 1 \
  --d_model 32 \
  --d_ff 128 \
  --batch_size 2 \
  --learning_rate 0.01 \
  --llm_layers 32 \
  --train_epochs 1 \
  --model_comment TimeLLM-ETTh1
