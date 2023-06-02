python train.py \
    --model_name_or_path google/flan-t5-base \
    --do_train \
    --do_eval \
    --train_file ./data/maybe/train.csv \
    --validation_file ./data/maybe/valid.csv \
    --source_prefix "Does the context require to search something on the Internet? User and bot utterances are split using \\n \n" \
    --output_dir ./model/train_on_maybe_valid \
    --text_column text \
    --summary_column label \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate