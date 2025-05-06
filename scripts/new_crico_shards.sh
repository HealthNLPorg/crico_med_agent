python /home/etg/Repos/CRICO/data_engineering/partition_time_by_job_count_shards.py \
       --input_tsv /home/etg/r/Eli/CRICO_106k/agent_2/242_16_28_remainder.tsv \
       --output_dir /home/etg/258_16_24_agent_2/ \
       --initial 258 \
       --job_count 16 \
       --hours_per_job 24 \
       --seconds_per_instance 35;
sudo rsync -ah --progress  /home/etg/258_16_24_agent_2/remainder.tsv /home/etg/r/Eli/CRICO_106k/agent_2/258_16_24_remainder.tsv
rm /home/etg/258_16_24_agent_2/remainder.tsv
scp -r /home/etg/258_16_24_agent_2 ch231037@e3-login.tch.harvard.edu:/home/ch231037/
