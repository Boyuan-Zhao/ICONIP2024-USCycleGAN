import os
import shutil
import json
import os.path as osp
import numpy as np
from sklearn.model_selection import train_test_split
from utils_ import Utils as utls

def copy2dst(imgs, root_path, organ, setname, label_info):
    trainA = utls.make_path(save_path, "{}A".format(setname))
    trainB = utls.make_path(save_path, "{}B".format(setname))
    for img in imgs:
        
        label_info[img] = labels.index(organ) + 1
        
        shutil.copy(osp.join(root_path, organ, "low_quality", img), trainA)
        shutil.copy(osp.join(root_path, organ, "high_quality", img), trainB)

def copy2dstv2(imgs, root_path, organ, setname, label_info):
    trainA = utls.make_path(save_path, "{}".format(setname), organ)
    trainB = utls.make_path(save_path, "{}".format(setname), organ)
    for img in imgs:
        
        label_info[img] = labels.index(organ) + 1
        
        shutil.copy(osp.join(root_path, organ, "low_quality", img), osp.join(trainA, img.replace(".", "_A.")))
        shutil.copy(osp.join(root_path, organ, "high_quality", img), osp.join(trainB, img.replace(".", "_B.")))

def splitTrainValv2():
    label_info = {}
    for organ in os.listdir(root_path):
        imgA = os.listdir(osp.join(root_path, organ, "low_quality"))
        
        np.random.seed(42)
        np.random.shuffle(imgA)
        train_img, val_img = utls.train_val_split(imgA, ratio_train=0.8, seed=42)

        copy2dstv2(train_img, root_path, organ, "train", label_info)
        copy2dstv2(val_img, root_path, organ, "val", label_info)


if __name__=="__main__":
    labels = ['breast', 'carotid', 'kidney', 'liver', 'thyroid']
    root_path = r"xxx/train_datasets"
    save_path = r"xxx/low2highv2"
    utls.create_dirs(save_path, rmtree=True)
    
    # 按照ImageNet格式随机8:2划分数据集
    splitTrainValv2()