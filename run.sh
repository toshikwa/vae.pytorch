# 'vae-123', 'vae-345', 'pvae'
model="pvae"
logdir="./log/${model}"
gpu="0"

python -u utils/train.py --model $model --logdir $logdir --gpu $gpu

exit 0