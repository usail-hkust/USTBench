 python run_UST_tasks.py --task socio_ecomic_prediction \
                         --batch_size 32 \
                         --llm_path_or_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                         --location "Guangzhou" \
                         --use_reflection false

 python run_UST_tasks.py --task congestion_prediction \
                         --batch_size 32 \
                         --llm_path_or_name .deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                         --location "Beijing" \
                         --use_reflection false

 python run_UST_tasks.py --task road_planning \
                         --batch_size 32 \
                         --llm_path_or_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                         --slum_name "CapeTown1" \
                         --use_reflection false

 python run_UST_tasks.py --task urban_planning \
                         --batch_size 32 \
                         --llm_path_or_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                         --cfg "hlg" \
                         --use_reflection false

python run_UST_tasks.py --task poi_placement \
                        --batch_size 32 \
                        --llm_path_or_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                        --location "Qiaonan" \
                        --use_reflection true

python run_UST_tasks.py --task traffic_signal_control \
                        --batch_size 32 \
                        --llm_path_or_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                        --dataset "hangzhou" \
                        --traffic_file "anon_4_4_hangzhou_real.json" \
                        --use_reflection true

python run_UST_tasks.py --task traffic_od_prediction \
                        --batch_size 32 \
                        --llm_path_or_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                        --location "Newyork" \
                        --use_reflection true

python run_UST_tasks.py --task route_planning \
                        --batch_size 32 \
                        --llm_path_or_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                        --location "Manhattan" \
                        --use_reflection true

python run_UST_tasks.py --task next_poi_prediction \
                        --batch_size 32 \
                        --llm_path_or_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                        --location "Newyork" \
                        --use_reflection true