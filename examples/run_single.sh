source /usr/local/anaconda3/bin/activate vlmeval
export CUDA_VISIBLE_DEVICES=2

# CITIES=('Beijing' 'Shanghai' 'SanFrancisco' 'NewYork' 'Mumbai' 'Tokyo' 'London' 'Paris' 'Moscow' 'SaoPaulo' 'Nairobi' 'CapeTown' 'Sydney')
# MODELS=('Yi_VL_6B' 'Yi_VL_34B' 'glm-4v-9b' 'InternVL2-2B' 'InternVL2-4B' 'InternVL2-8B' 'InternVL2-26B')
CITIES=('Beijing')
MODELS=('Qwen2-VL-2B-Instruct')
DATA_VERSION='mini'

for MODEL in "${MODELS[@]}"; do
    echo "Current model: $MODEL"
    for CITY in "${CITIES[@]}"; do
        echo "Current city: $CITY"
        python -m citybench.remote_sensing.eval_inference --task_name="population" --city_name=$CITY --model_name=$MODEL --data_version=$DATA_VERSION
    done
done
