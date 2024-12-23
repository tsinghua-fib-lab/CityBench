source /usr/local/anaconda3/bin/activate vlmeval2
export CUDA_VISIBLE_DEVICES=2

CITIES=('Beijing' 'SanFrancisco' 'NewYork' 'Mumbai' 'Tokyo' 'London' 'Paris' 'Moscow' 'SaoPaulo' 'Nairobi' 'CapeTown' 'Sydney' 'Shanghai')
# MODELS=('Qwen2-VL-2B-Instruct' 'Qwen2-VL-7B-Instruct' 'Yi_VL_6B' 'glm-4v-9b' 'InternVL2-2B' 'InternVL2-4B' 'InternVL2-8B')
# MODELS=('InternVL2-26B' 'Yi_VL_34B' 'InternVL2-40B')
MODELS=('glm-4v-9b')
DATA_VERSION='all'

for MODEL in "${MODELS[@]}"; do
    echo "Current model: $MODEL"
    for CITY in "${CITIES[@]}"; do
        echo "Current city: $CITY"
        python -m citybench.street_view.eval_inference --task_name="geoloc" --city_name=$CITY --model_name=$MODEL --data_version=$DATA_VERSION
    done
done
