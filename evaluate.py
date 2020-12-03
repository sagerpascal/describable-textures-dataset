import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from dataset import DtdDataset, get_preprocessing
from meter import AverageValueMeter
from metrics import AccuracyT1

# Config
# Read Config
conf = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
device = conf['device']
num_classes = conf['num_classes']

# Parameters from args
parser = argparse.ArgumentParser()
parser.add_argument("--model_name")
parser.add_argument("--accuracy")
parser.add_argument("--plot")

args = parser.parse_args()
model_name = str(args.model_name)
calc_accuracy = bool(args.accuracy)
plot_examples = bool(args.plot)


def load_model():
    model = torch.load("trained_models/{}.pth".format(model_name), map_location=torch.device(device))
    model.eval()
    mask_size = model.get_mask_size() if model_name.startswith('simple_fcn') else (128, 128)
    return model, mask_size


def get_dataset(mask_size):
    return DtdDataset('data/dtd_test_tiled', mask_size=mask_size, augmentation=None, preprocessing=get_preprocessing())


def unormalize_image(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def plot_some_examples(dataset, model, n_examples):
    for i in tqdm(range(n_examples)):
        img, mask, _ = dataset[i]

        if device == 'cuda':
            img, mask = img.cuda(), mask.cuda()

        X = img.unsqueeze(0)
        pred = model.predict(X)

        amask = mask.cpu().numpy().argmax(axis=0)
        pmask = pred.cpu().numpy()[0].argmax(axis=0)
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = unormalize_image(img)

        fig, axs = plt.subplots(ncols=3, figsize=(10, 4))

        axs[0].set_title("Image")
        axs[0].imshow(img)
        axs[1].set_title("Predicted Mask")
        axs[1].imshow(pmask, vmin=0, vmax=num_classes-1, cmap='prism')
        axs[2].set_title("Actual Mask")
        axs[2].imshow(amask, vmin=0, vmax=num_classes-1, cmap='prism')
        plt.show()


def get_accuracy(dataset, model):
    acc_meter = AverageValueMeter()
    acc_fn = AccuracyT1()

    with tqdm(dataset, desc='Calculating Top-1 Accuracy', file=sys.stdout) as iterator:
        for x, y, _ in iterator:

            y = y.unsqueeze(axis=0)
            if device == 'cuda':
                x, y = x.cuda(), y.cuda()

            X = x.unsqueeze(0)
            y_pred = model.predict(X)

            acc_value = acc_fn(y_pred, y).cpu().detach().numpy()
            acc_meter.add(acc_value)
            iterator.set_postfix_str("Accuracy - {}".format(acc_meter.mean))

    return acc_meter.mean


def main():
    model, mask_size = load_model()
    dataset = get_dataset(mask_size)
    if plot_examples:
        plot_some_examples(dataset, model, n_examples=30)
    if calc_accuracy:
        print("Accuracy of model: {}".format(get_accuracy(dataset, model)))


if __name__ == '__main__':
    main()
