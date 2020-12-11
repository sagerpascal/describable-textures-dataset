import os

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import yaml

# Read Config
conf = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
num_classes = conf['num_classes']
loss = conf['loss']
tiled = conf['tiled']

# Mask range: 0-46
class DtdDataset(torch.utils.data.Dataset):

    def __init__(self, image_path, mask_size, augmentation=None, preprocessing=None):

        self.image_prefix = 'image_'
        self.mask_prefix = 'label_'

        self.files = []
        for file in os.listdir(image_path):
            if file.startswith(self.image_prefix):
                fn = file.split(self.image_prefix, 1)[1]
                if os.path.exists("{}/{}{}".format(image_path, self.mask_prefix, fn)):
                    self.files.append(fn)
                else:
                    print("Mask not found for '{}'".format(fn))
            else:
                continue
        print("Loaded {} files".format(len(self.files)))

        self.image_path = image_path
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.mask_size = mask_size

    def __getitem__(self, item):

        filename = self.files[item]
        img = Image.open(os.path.join(self.image_path, self.image_prefix + filename))
        img = img.convert("RGB") if tiled else img.convert("L")
        mask = Image.open(os.path.join(self.image_path, self.mask_prefix + filename))
        img = np.array(img)
        mask = np.array(mask)

        mask = self.normalize_mask(mask)
        mask = self.resize_mask(mask)
        img = self.normalize_image(img, filename)

        # apply augmentation
        if self.augmentation is not None:
            sample = self.augmentation(image=img)
            img = sample['image']

        label = np.array(mask[0][0])
        # check if all labels are equal
        assert np.all(mask == label)

        # apply preprocessing
        if self.preprocessing is not None:
            sample = self.preprocessing(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        return img, mask, label

    def resize_mask(self, mask):
        y, x = mask.shape
        startx = x // 2 - (self.mask_size[0] // 2)
        starty = y // 2 - (self.mask_size[1] // 2)
        return mask[starty:starty + self.mask_size[1], startx:startx + self.mask_size[0]]

    def normalize_mask(self, mask):
        return mask - 1

    def normalize_image(self, image, filename):
        img = image / 255.0
        return (img - np.mean(img, axis=(0, 1), keepdims=True)) / np.std(img, axis=(0, 1), keepdims=True)

    def __len__(self):
        return len(self.files)


def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=1.),
        # A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1., border_mode=0),
        #
        # A.PadIfNeeded(min_height=128, min_width=128, always_apply=True, border_mode=0),
        # A.RandomCrop(height=128, width=128, always_apply=True),
        #
        # A.IAAAdditiveGaussianNoise(p=0.2),
        # A.IAAPerspective(p=0.5),
        #
        # A.OneOf(
        #     [
        #         A.CLAHE(p=1.),
        #         A.RandomBrightness(p=1.),
        #         A.RandomGamma(p=1.),
        #     ],
        #     p=0.9,
        # ),
        #
        # A.OneOf(
        #     [
        #         A.IAASharpen(p=1.),
        #         A.Blur(blur_limit=3., p=1.),
        #         A.MotionBlur(blur_limit=3., p=1.),
        #     ],
        #     p=0.9,
        # ),
        #
        # A.OneOf(
        #     [
        #         A.RandomContrast(p=1.),
        #         A.HueSaturationValue(p=1.),
        #     ],
        #     p=0.9,
        # ),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    valid_transform = [
        A.PadIfNeeded(128, 128)
    ]
    return A.Compose(valid_transform)


def to_tensor(x, **kwargs):
    '''
    Convert image to tensor
    '''
    if tiled:
        return torch.as_tensor(x.transpose(2, 0, 1).astype('float32'))
    else:
        return torch.as_tensor(x.astype('float32')).unsqueeze(dim=0)


def to_tensor_mask(x, **kwargs):
    '''
    Convert mask to tensor
    '''
    if loss == 'cross-entropy':
        return torch.as_tensor(x.astype('int64'))
    else:
        x_tensor = torch.as_tensor(x.astype('int64'))
        x_oh = torch.nn.functional.one_hot(x_tensor, num_classes=num_classes).type(torch.DoubleTensor)
        return x_oh.permute(2, 0, 1)


def get_preprocessing():
    '''
    Preprocess the image and the mask
    '''
    _transform = [
        A.Lambda(image=to_tensor, mask=to_tensor_mask),
    ]
    return A.Compose(_transform)


def plot_some_data():
    def unormalize_image(img):
        return (img - np.min(img)) / (np.max(img) - np.min(img))

    dataset = DtdDataset('data/dtd_train_tiled', mask_size=(128, 128), preprocessing=get_preprocessing())

    for i in range(len(dataset)):
        img, mask, lab = dataset[i]
        if lab == 0:
            img = img.numpy().transpose(1, 2, 0)
            img = unormalize_image(img)
            amask = mask.numpy().argmax(axis=0)

            fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
            axs[0].set_title("Image")
            axs[0].imshow(img)
            axs[1].set_title("Mask (Label: {})".format(lab))
            axs[1].imshow(amask, vmin=0, vmax=num_classes - 1, cmap='prism')
            plt.show()


if __name__ == '__main__':
    plot_some_data()
