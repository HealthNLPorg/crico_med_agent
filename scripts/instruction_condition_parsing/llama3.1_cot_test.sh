python ~/Repos/cnlp_transformers/src/cnlpt/cnlp_seqgen.py \
       --model_path unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
       --prompt_file ~/CRICO_experiments_packet_PHI/first_edit_example_in_message_format.txt \
       --examples_file ~/CRICO_experiments_packet_PHI/cot_example.txt \
       --load_in_4bit \
       --max_new_tokens 2048 \
       --query_files ~/CRICO_experiments_packet_PHI/CRICO_windowed/subsample_core_train_notes.tsv \
       --output_dir ~/CRICO_experiments_packet_PHI/temp_result/ \
       --fancy_output
