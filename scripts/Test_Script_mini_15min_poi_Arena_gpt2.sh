# Set number of processes (1 per GPU)
num_process=2
master_port=29500  # or any free port
#seq_len -> 24*7 = 168 (24h for 7days = 1 week)
#pred_len -> 24*4 = 96 or predict the next 4 h
# Configure Accelerate for Kaggle (might require setup beforehand)
#accelerate launch --num_processes $num_process --main_process_port $master_port run_main.py \
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/final_15min_df_poi/ \
  --data_path frequency_15min_df_49_year2019.csv \
  --model_id Arena_Verona_2019 \
  --model TimeLLM \
  --llm_model GPT2 \
  --llm_dim 768 \
  --data Arena_Verona_2019_mini2 \
  --features S \
  --target frequency \
  --freq 15min \
  --scale True \
  --seq_len 192 \
  --label_len 96 \
  --pred_len 48 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --prompt_domain 1 \
  --moving_avg 96 \
  --des Exp \
  --itr 1 \
  --d_model 32 \
  --d_ff 128 \
  --n_head 4 \
  --batch_size 15 \
  --eval_batch_size 6 \
  --learning_rate 0.01 \
  --llm_layers 8 \
  --train_epochs 10 \
  --seasonal_patterns Daily \
  --model_comment TimeLLM_Arena_Verona_2019_15min_1epochs
