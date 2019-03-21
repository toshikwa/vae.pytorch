gpu="0"
attr="Bald"

for model in 'vae-123' 'vae-345' 'pvae'; do
    logdir="./log/${model}"
    path="${logdir}/final_model.pth"
    python -u utils/interpolate.py --model $model --logdir $logdir --gpu $gpu --attr $attr --path $path
done

exit 0