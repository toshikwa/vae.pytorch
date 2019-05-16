# Deep Feature Consistent Variational Autoencoder in PyTorch
A PyTorch implementation of Deep Feature Consistent Variational Autoencoder. I implemented DFC-VAE based on [the paper](https://arxiv.org/pdf/1610.00291.pdf) by Xianxu Hou, Linlin Shen, Ke Sun, Guoping Qiu. I trained this model with CelebA dataset. For more details about the dataset, please refer to [the website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

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


## Interpolate
- Use command below.
```
sh interpolate.sh
```


## ToDo
- Add experiments with other datasets(with more large image size).
