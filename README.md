# Deep Feature Consistent Variational Autoencoder in PyTorch
A PyTorch implementation of Deep Feature Consistent Variational Autoencoder. Our DFC-VAE is based on [the paper](https://arxiv.org/pdf/1610.00291.pdf) by Xianxu Hou, Linlin Shen, Ke Sun, Guoping Qiu. I trained this model with CelebA dataset. For more details about the dataset, please refer to [the website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

## Installation
- Clone this repository.
- python 3.6 is recommended.
- Use command `pip install -r requirements.txt` to install libraries.

## Dataset
- You need to download the CelebA dataset from [the website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and arrange them like below.
```
.
├── README.md
├── requirements.txt
├── models
├── utils
...
└── data
    └── celeba
        ├── images
        │   ├── 000001.jpg
        │   └── ...
        └── annotations
            ├── list_attr_celeba.txt
            └── ...
```

## Train
- Use command below.
```
sh run.sh
```

Examples after training 5 epochs.

### pvae
![pvae](https://github.com/ku2482/vae.pytorch/blob/master/sample/pvae.png)

### vae-123
![vae-123](https://github.com/ku2482/vae.pytorch/blob/master/sample/vae-123.png)

### vae-345
![vae-345](https://github.com/ku2482/vae.pytorch/blob/master/sample/vae-345.png)

## Interpolate
- Use command below.
```
sh interpolate.sh
```


Examples interpolating between non-bald and bald images.

### pvae
![pvae](https://github.com/ku2482/vae.pytorch/blob/master/sample/pvae-bald.png)

### vae-123
![vae-123](https://github.com/ku2482/vae.pytorch/blob/master/sample/vae-123-bald.png)

### vae-345
![vae-345](https://github.com/ku2482/vae.pytorch/blob/master/sample/vae-345-bald.png)


## ToDo
- Add experiments with other datasets(with more large image size).
