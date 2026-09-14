from __future__ import print_function, division
import os
import cv2
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr


class Segmentation(Dataset):
    NUM_CLASSES = 2

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('Kvasir'),
                 split='train',
                 ):
        """
        :param base_dir: path to dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = Path.db_root_dir(args.dataset)
        self._image_dir = os.path.join(self._base_dir, 'images')
        self._cat_dir = os.path.join(self._base_dir, 'masks')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        _splits_dir = os.path.join(self._base_dir)

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            # self._image_dir = os.path.join(self._base_dir, splt, 'images')
            # self._cat_dir = os.path.join(self._base_dir, splt, 'masks')
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + '.jpg')
                _cat = os.path.join(self._cat_dir, line + '.jpg')
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))


    def __len__(self):
        if self.split[0] == 'test':
            return len(self.images)
        else:
            return len(self.images) // self.args.batch_size * self.args.batch_size

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample), self.im_ids[index]
            elif split == 'test':
                return self.transform_test(sample), self.im_ids[index]

    def _make_img_gt_point_pair(self, index):
        _img = cv2.imread(self.images[index], 1)
        _target = cv2.imread(self.categories[index], 0)

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHueSaturationValue(hue_shift_limit=(-30, 30),
                                        sat_shift_limit=(-5, 5),
                                        val_shift_limit=(-15, 15)),
            tr.RandomShiftScaleRotate(shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0)),
            tr.RandomHorizontalFlip(),
            tr.RandomVerticalFlip(),
            tr.RandomRotate(90),
            tr.Normalize(image_size=self.args.image_size),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.Normalize(image_size=self.args.image_size),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_test(self, sample):
        composed_transforms = transforms.Compose([
            tr.Normalize_test(image_size=self.args.image_size),
            tr.ToTensor_test()
        ])

        return composed_transforms(sample)
