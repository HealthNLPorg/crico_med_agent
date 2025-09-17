# ~/r/Eli/CRICO_106k/agent_2/finished/284_16_20_agent_2/ \
# ~/r/Eli/CRICO_106k/agent_2/finished/300_15_2_agent_2/ \
# ~/r/Eli/CRICO_106k/agent_2/finished/315_16_24_agent_2/ \
# ~/r/Eli/CRICO_106k/agent_2/finished/331_16_24_agent_2/ \
# ~/r/Eli/CRICO_106k/agent_2/finished/347_16_24_agent_2/ \
# ~/r/Eli/CRICO_106k/agent_2/finished/363_16_48_agent_2/ \
# ~/r/Eli/CRICO_106k/agent_2/finished/379_16_72_agent_2/ \
# ~/r/Eli/CRICO_106k/agent_2/finished/395_10_72_agent_2/ \
# ~/r/Eli/CRICO_106k/agent_2/finished/405_6_72_agent_2/ \
# Difference with this one is I had to rename the last shard for the
# script to work -  bc I didn't name it shard_frame.tsv
python ~/Repos/CRICO/data_engineering/reassemble_shards.py \
  --input_folders ./411_17_9_agent_2 \
  --output_dir ~/NORMAL_ASSEMBLY_merged_shards_411
