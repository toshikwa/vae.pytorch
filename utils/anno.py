import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

celeba_dir = os.path.join(".", "data", "celeba")
anno_dir = os.path.join(celeba_dir, "annotations")
img_dir = os.path.join(celeba_dir, "images")

def parse_annotation(path):
    with open(path, 'r') as f:
        texts = f.read().split("\n")[1:]

    columns = np.array(texts[0].split(" "))
    columns = columns[columns != ""]

    df = []
    for txt in texts[1:]:
        txt = np.array(txt.split(" "))
        txt = txt[txt!= ""]
        df.append(txt)

    df = pd.DataFrame(df)

    if df.shape[1] == len(columns) + 1:
        columns = ["image_id"] + list(columns)
    df.columns = columns

    df = df.dropna()
    for nm in df.columns:
        if nm != "image_id":
            df[nm] = pd.to_numeric(df[nm],downcast="integer")

    return df

def get_df(attr):
    df_path = os.path.join(anno_dir, "list_attr_celeba.csv")

    if not os.path.exists(df_path):
        df = parse_annotation(os.path.join(anno_dir, "list_attr_celeba.txt"))
        df.to_csv(df_path, index=False)
    else:
        df = pd.read_csv(df_path)
    return df[["image_id", attr]]

def gen_images(attr, num=100):
    df = get_df(attr)
    
    pos_images = df[df[attr] == 1].image_id.values[:num]
    neg_images = df[df[attr] == -1].image_id.values[:num]

    output = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        ])
    
    for p, n in zip(pos_images, neg_images):
        p_ = Image.open(os.path.join(img_dir, p))
        n_ = Image.open(os.path.join(img_dir, n))
        yield output(p_), output(n_)
    