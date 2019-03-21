gpu="0"
model="vae-345"
logdir="./log/${model}"

python -u utils/train.py --model $model --logdir $logdir --gpu $gpu