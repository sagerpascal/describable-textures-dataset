import argparse

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


from dataset import DtdDataset, get_training_augmentation, get_validation_augmentation
from epoch import TrainEpoch, ValidEpoch
from metrics import Accuracy, Fscore, Recall, Precision, AccuracyT1
from models import SimpleFullyCnn, SimpleUNet

# Config
device = 'cpu'
num_classes = 47
num_epoch = 75
model_name = 'Simple U-Net'
use_wandb = False

# Parameters from command line
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate")
parser.add_argument("--batch_size")
parser.add_argument("--pweight_factor")

args = parser.parse_args()
lr = float(args.learning_rate)  # 0.000005
batch_size = int(args.batch_size)  # 8
pos_weight_factor = int(args.pweight_factor)

torch.cuda.empty_cache()


def to_tensor(x, **kwargs):
    x = np.expand_dims(x, axis=0)
    return torch.as_tensor(x.astype('float32'))


def to_tensor_mask(x, **kwargs):
    # With BCELossWithDigits
    x_tensor = torch.as_tensor(x.astype('int64'))
    x_oh = torch.nn.functional.one_hot(x_tensor, num_classes=num_classes).type(torch.DoubleTensor)
    return x_oh.permute(2, 0, 1)
    # With CrossEntropyLoss
    # return torch.as_tensor(x.astype('int64'))


def get_preprocessing():
    _transform = [
        A.Lambda(image=to_tensor, mask=to_tensor_mask),
    ]
    return A.Compose(_transform)


def get_data_loaders(mask_size):
    train_dataset = DtdDataset('data/dtd_train', mask_size, augmentation=get_training_augmentation(),
                               preprocessing=get_preprocessing())
    valid_dataset = DtdDataset('data/dtd_val', mask_size, augmentation=get_validation_augmentation(),
                               preprocessing=get_preprocessing())
    test_dataset = DtdDataset('data/dtd_test', mask_size, augmentation=None, preprocessing=get_preprocessing())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    return train_loader, valid_loader, test_loader


class MyWeightedLoss(torch.nn.BCEWithLogitsLoss):

    def __init__(self, mask_size, **kwargs):
        pos_weight = np.ones(
            (num_classes, mask_size[0], mask_size[1])) * pos_weight_factor  # increase recall (for the cost of a worse precision)
        pos_weight = torch.FloatTensor(pos_weight)
        super().__init__(pos_weight=pos_weight, **kwargs)

    def forward(self, y_pr, y_gt):
        return super(MyWeightedLoss, self).forward(y_pr, y_gt)


def main():
    import wandb
    if use_wandb:
        wandb.init(project="compvis_dtd_{}".format(model_name))
        wandb.config.update({"Model": model_name,
                             "Learning Rate": lr,
                             "Batch Size": batch_size,
                             "Pos- Weight": pos_weight_factor
                             })
    else:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()

    if model_name == 'Simple FCN':
        model = SimpleFullyCnn(in_channels=1, out_channels=num_classes).to(device)
        mask_size = model.get_mask_size()

    elif model_name == 'Simple U-Net':
        model = SimpleUNet(input_channels=1, output_channels=num_classes).to(device)
        mask_size = (128, 128)

    elif model_name == 'U-Net pretrained encoder':
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name='resnet101',
            encoder_weights='imagenet',
            classes=num_classes,
            in_channels=1,
            decoder_use_batchnorm=True,
            activation='softmax2d'
        )
        mask_size = (128, 128)

    else:
        raise NotImplementedError("no such model: {}".format(model_name))

    train_loader, valid_loader, _ = get_data_loaders(mask_size)

    loss = MyWeightedLoss(mask_size)
    loss.__name__ = 'bce-with-logits-loss'

    # loss = torch.nn.CrossEntropyLoss()
    # loss.__name__ = 'Cross Entropy Loss'

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    metrics = [
        Accuracy(),
        AccuracyT1(),
        Fscore(),
        Recall(),
        Precision(),
    ]

    train_epoch = TrainEpoch(model, loss=loss, metrics=metrics, optimizer=optimizer, device=device, verbose=True)
    valid_epoch = ValidEpoch(model, loss=loss, metrics=metrics, device=device, verbose=True)

    best_loss = 9999
    count_not_improved = 0
    for i in range(num_epoch):
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        if best_loss > valid_logs[loss.__name__] + 0.00005:
            best_loss = valid_logs[loss.__name__]
            if i > 10:
                if use_wandb:
                    torch.save(model, '{}-best_model-{}.pth'.format(wandb.run.name, i))
                else:
                    torch.save(model, '{}-best_model-{}.pth'.format(model_name, i))
                print("Model saved")
            count_not_improved = 0
        else:
            count_not_improved += 1

        if use_wandb:
            logs = {"epoch": i, "train loss": train_logs[loss.__name__], "valid loss": valid_logs[loss.__name__]}
            for m in metrics:
                m_name = m.__name__
                logs["{} train".format(m_name)] = train_logs[m_name]
                logs["{} valid".format(m_name)] = valid_logs[m_name]
            wandb.log(logs)
        else:
            writer.add_scalar('{}/Loss/train'.format(model_name), train_logs[loss.__name__], i)
            writer.add_scalar('{}/Loss/valid'.format(model_name), valid_logs[loss.__name__], i)

            for m in metrics:
                m_name = m.__name__
                writer.add_scalar('{}/{}/train'.format(model_name, m_name), train_logs[m_name], i)
                writer.add_scalar('{}({}/valid'.format(model_name, m_name), valid_logs[m_name], i)

        # Early stopping
        if count_not_improved > 8:
            print("Early stopping!")
            break


def analyze_masks():
    train_dataset = DtdDataset('images/dtd_train', (128, 128), augmentation=None, preprocessing=None)

    labels = []
    for i in range(len(train_dataset)):
        _, _, label = train_dataset[i]
        labels.append(label)

    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    dict(zip(unique, counts))

    fig = plt.figure(figsize=(8, 8))
    width = 0.65
    plt.bar(unique, counts, width)
    plt.ylabel('Number of Mask')
    plt.title('Number of Masks per Category')
    plt.xticks(unique)
    plt.yticks(np.arange(0, 51, 10))
    plt.show()


if __name__ == '__main__':
    # analyze_masks()
    main()
