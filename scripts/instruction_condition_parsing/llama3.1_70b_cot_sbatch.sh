#!/bin/bash
# Sample batchscript to run a GPU-Cuda job on HPC
#SBATCH --account=chip
#SBATCH --partition=chip-gpu             # queue to be used
#SBATCH --time=96:00:00             # Running time (in hours-minutes-seconds)
#SBATCH --job-name=llama3.1_70b_cot            # Job name
#SBATCH --mail-user=eli.goldner@childrens.harvard.edu      # Email address to send the job status
#SBATCH --output=/home/ch231037/logs/log_%x_%j.txt          # Name of the output file
#SBATCH --nodes=1               # Number of gpu nodes
#SBATCH --ntasks=1               # Number of gpu nodes
#SBATCH --gres=gpu:NVIDIA_A40:1                # Number of gpu devices on one gpu node
#SBATCH --mem=16GB

source /home/ch231037/.bashrc
# conda init
conda activate hf_latest

cd ~
python ~/Repos/cnlp_transformers/src/cnlpt/cnlp_seqgen.py \
       --model_path unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit \
       --prompt_file ~/CRICO_experiments_packet_PHI/first_edit_example_in_message_format.txt \
       --examples_file ~/CRICO_experiments_packet_PHI/cot_example.txt \
       --load_in_4bit \
       --max_new_tokens 2048 \
       --query_files ~/CRICO_experiments_packet_PHI/CRICO_windowed/subsample_core_train_notes.tsv \
       --output_dir ~/CRICO_experiments_packet_PHI/temp_result/ \
       --fancy_output
