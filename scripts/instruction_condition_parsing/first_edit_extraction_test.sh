python ~/Repos/cnlp_transformers/src/cnlpt/cnlp_seqgen.py --model_path unsloth/llama-3-8b-Instruct-bnb-4bit \
       --prompt_file ~/CRICO_experiments_packet_PHI/first_edit_with_built_in_example.txt \
       --load_in_4bit \
       --max_new_tokens 1024 \
       --query_files ~/CRICO_experiments_packet_PHI/CRICO_windowed/subsample_core_train_notes.tsv \
       --output_dir ~/CRICO_experiments_packet_PHI/temp_result/ \
       --fancy_output

