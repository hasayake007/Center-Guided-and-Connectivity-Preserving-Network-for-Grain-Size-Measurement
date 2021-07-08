import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class grain_Loader(Dataset):
    def __init__(self, data_path, mode):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.img_path = glob.glob(os.path.join(data_path, 'img/*.JPG'))
        self.mode = mode
        if self.mode == 'train':
            self.transform = A.Compose([
                A.RandomScale(scale_limit=0.2, p=0.5),
                A.PadIfNeeded(min_height=512, min_width=512),
                A.CenterCrop(512, 512),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose([
                A.Resize(2336, 2336),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
                ]
             )

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.img_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('img', 'label')
        label_path = label_path.replace('.JPG', '.png')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        label = np.where(label > 0, 1, 0)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]

        mask = label.view(1, label.shape[0], label.shape[1])
        mask = mask.cpu().numpy()
        mask = mask.astype('float64')
        mask_128 = cv2.resize(mask[0, :, :], (128, 128), interpolation=cv2.INTER_NEAREST)
        mask, skeleton = torch.from_numpy(mask), torch.from_numpy(mask_128)
        skeleton = skeleton.unsqueeze(dim=0)
        return image, mask.float(), skeleton.float()

    def __len__(self):
        # 返回训练集大小
        return len(self.img_path)


