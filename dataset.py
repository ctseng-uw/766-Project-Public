import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import random


class FontToFontDataset:
    def __init__(
        self,
        src_font_dir: str,
        dst_font_dir: str,
        shuffle: bool,
        random_flip: bool,
        to_gray: bool,
    ):
        self.src_font_dir = src_font_dir
        self.dst_font_dir = dst_font_dir
        self.random_flip = random_flip
        self.to_gray = to_gray

        # List all images name to speed up
        self.src_font_imgs_names = sorted(os.listdir(src_font_dir))
        self.dst_font_imgs_names = sorted(os.listdir(dst_font_dir))

        assert len(self.src_font_imgs_names) == len(self.dst_font_imgs_names)
        self.length_dataset = len(self.dst_font_imgs_names)

        self.src_idx = list(range(self.length_dataset))
        self.dst_idx = list(range(self.length_dataset))

        if shuffle:
            random.shuffle(self.src_idx)
            random.shuffle(self.dst_idx)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        src_font_img_path = os.path.join(
            self.src_font_dir, self.src_font_imgs_names[self.src_idx[index]]
        )
        dst_font_img_path = os.path.join(
            self.dst_font_dir, self.dst_font_imgs_names[self.dst_idx[index]]
        )

        src_font_img_path2 = os.path.join(
            self.src_font_dir, self.src_font_imgs_names[self.dst_idx[index]]
        )
        dst_font_img_path2 = os.path.join(
            self.dst_font_dir, self.dst_font_imgs_names[self.src_idx[index]]
        )

        src_font_img = np.array(Image.open(src_font_img_path))
        dst_font_img = np.array(Image.open(dst_font_img_path))

        src_font_img2 = np.array(Image.open(src_font_img_path2))
        dst_font_img2 = np.array(Image.open(dst_font_img_path2))

        src_font_img = T.ToTensor()(src_font_img)
        dst_font_img = T.ToTensor()(dst_font_img)

        src_font_img2 = T.ToTensor()(src_font_img2)
        dst_font_img2 = T.ToTensor()(dst_font_img2)

        if self.to_gray:
            src_font_img = T.Grayscale()(src_font_img)
            dst_font_img = T.Grayscale()(dst_font_img)
            src_font_img2 = T.Grayscale()(src_font_img2)
            dst_font_img2 = T.Grayscale()(dst_font_img2)

        if self.random_flip:
            flip1h = random.choice([True, False])
            flip2h = random.choice([True, False])
            flip1v = random.choice([True, False])
            flip2v = random.choice([True, False])
            if flip1h:
                src_font_img = T.RandomHorizontalFlip(p=1.0)(src_font_img)
                dst_font_img2 = T.RandomHorizontalFlip(p=1.0)(dst_font_img2)
            if flip2h:
                dst_font_img = T.RandomHorizontalFlip(p=1.0)(dst_font_img)
                src_font_img2 = T.RandomHorizontalFlip(p=1.0)(src_font_img2)
            if flip1v:
                src_font_img = T.RandomVerticalFlip(p=1.0)(src_font_img)
                dst_font_img2 = T.RandomVerticalFlip(p=1.0)(dst_font_img2)
            if flip2v:
                dst_font_img = T.RandomVerticalFlip(p=1.0)(dst_font_img)
                src_font_img2 = T.RandomVerticalFlip(p=1.0)(src_font_img2)
        return src_font_img, dst_font_img, src_font_img2, dst_font_img2


def get_dataloader(
    src_dir: str,
    dst_dir: str,
    batch_size: int,
    shuffle: bool,
    random_flip: bool,
    to_gray: bool,
):
    dataset = FontToFontDataset(src_dir, dst_dir, shuffle, random_flip, to_gray)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
