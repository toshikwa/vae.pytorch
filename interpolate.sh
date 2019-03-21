gpu="0"
attr="Smiling"

for model in 'vae-123' 'vae-345' 'pvae'; do
    logdir="./log/${model}"
    python -u utils/interpolate.py --model $model --logdir $logdir --gpu $gpu --attr $attr
done

exit 0