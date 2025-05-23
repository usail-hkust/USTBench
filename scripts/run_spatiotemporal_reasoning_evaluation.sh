python run_UST_tasks.py --task question_answering \
                        --batch_size 32 \
                        --llm_path_or_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                        --tasks "next_poi_prediction, congestion_prediction, socio_economic_prediction, traffic_od_prediction" \
                        --datasets "st_understanding, forecasting, reflection"

python run_UST_tasks.py --task question_answering \
                        --batch_size 32 \
                        --llm_path_or_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                        --tasks "traffic_signal_control, poi_placement, road_planning, route_planning, urban_planning" \
                        --datasets "st_understanding, planning, reflection"