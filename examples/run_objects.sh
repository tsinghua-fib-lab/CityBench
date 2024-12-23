source /usr/local/anaconda3/bin/activate vlmeval2
export CUDA_VISIBLE_DEVICES=2

# 去除Beijing，Shanghai
CITIES=('SanFrancisco' 'NewYork' 'Mumbai' 'Tokyo' 'London' 'Paris' 'Moscow' 'SaoPaulo' 'Nairobi' 'CapeTown' 'Sydney')
# CITIES=('SaoPaulo')
# MODELS=('Qwen2-VL-2B-Instruct' 'Qwen2-VL-7B-Instruct' 'Yi_VL_6B' 'glm-4v-9b' 'InternVL2-2B' 'InternVL2-4B' 'InternVL2-8B')
MODELS=('InternVL2-26B' 'Yi_VL_34B' 'InternVL2-40B')
DATA_VERSION='all'

for MODEL in "${MODELS[@]}"; do
    echo "Current model: $MODEL"
    for CITY in "${CITIES[@]}"; do
        echo "Current city: $CITY"
        python -m citybench.remote_sensing.eval_inference --task_name="objects" --city_name=$CITY --model_name=$MODEL --data_version=$DATA_VERSION
    done
done
