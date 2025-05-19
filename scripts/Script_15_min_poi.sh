# Set number of processes (1 per GPU)
num_process=2
master_port=29500  # or any free port
#seq_len -> 4 = 15min *4 = 1h / seq_len -> 96 = 1 giorno (4fasce(15min) * 24ore)
#pred_len -> 1 = 15min / 4 = next hour
# Configure Accelerate for Kaggle (might require setup beforehand)
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/final_15min_df_poi/ \
  --data_path frequency_15min_df_61.csv \
  --model_id Giulietta \
  --model TimeLLM \
  --data Casa_Giulietta \
  --features S \
  --target frequency \
  --freq 15min \
  --seq_len 96 \
  --label_len 24 \
  --pred_len 1 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --patch_len 30 \
  --prompt_domain 1 \
  --moving_avg 96 \
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
  --model_comment TimeLLM-Casa-Giulietta-15min