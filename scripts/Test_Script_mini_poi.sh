# Set number of processes (1 per GPU)
num_process=2
master_port=29500  # or any free port
#seq_len -> 24*7 = 168 (24h for 7days = 1 week)
#pred_len -> 24*4 = 96 or predict the next 4 h
# Configure Accelerate for Kaggle (might require setup beforehand)
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/final_df_poi/ \
  --data_path freq_mini.csv \
  --model_id freq_mini_df_42 \
  --model TimeLLM \
  --data Traffic \
  --features S \
  --target frequency \
  --freq h \
  --seq_len 168 \
  --label_len 24 \
  --pred_len 4 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --moving_avg 7 \
  --des Exp \
  --itr 1 \
  --d_model 32 \
  --d_ff 128 \
  --n_head 4 \
  --batch_size 8 \
  --eval_batch_size 8 \
  --learning_rate 0.01 \
  --llm_layers 8 \
  --train_epochs 1 \
  --model_comment TimeLLM_mini_poi_42
