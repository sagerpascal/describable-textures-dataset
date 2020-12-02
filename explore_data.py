from tqdm import tqdm
import numpy as np
from dataset import DtdDataset
import matplotlib.pyplot as plt


def analyze_masks():
    '''
    Plot the mask distribution
    '''
    train_dataset = DtdDataset('data/dtd_train_tiled', (128, 128), augmentation=None, preprocessing=None)

    labels = []
    for i in tqdm(range(len(train_dataset)), desc='Analyzing data'):
        _, _, label = train_dataset[i]
        labels.append(label)

    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    dict(zip(unique, counts))

    fig = plt.figure(figsize=(16, 8))
    width = 0.65
    bars = plt.bar(unique, counts, width)
    plt.ylabel('Number of Mask')
    plt.title('Number of Masks per Category')
    plt.xticks(unique)
    plt.yticks(np.arange(0, 601, 10))

    for b in bars:
        height = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2., 1.05 * height,
                 '%d' % int(height),
                 ha='center', va='bottom')

    plt.show()


if __name__ == '__main__':
    analyze_masks()
