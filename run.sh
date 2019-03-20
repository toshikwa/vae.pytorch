gpu="0"
model="pvae"
logdir="./log/${model}"

python -u utils/train.py --model $model --logdir $logdir --gpu $gpu