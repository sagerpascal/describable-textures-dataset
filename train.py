import argparse
import os

import wandb
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import DtdDataset, get_preprocessing
from epoch import TrainEpoch, ValidEpoch
from losses import WeightedBCEWithLogitsLoss, DiceLoss
from metrics import Accuracy, Fscore, Recall, Precision, AccuracyT1
from models import SimpleFullyCnn, SimpleUnet

# Read Config
conf = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
num_classes = conf['num_classes']
device = conf['device']
tiled = conf['tiled']
loss = conf['loss']
max_num_epoch = conf['max_num_epoch']

# Read parameters from command line
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate")
parser.add_argument("--batch_size")
parser.add_argument("--model_name")
parser.add_argument("--wandb")
parser.add_argument("--early_stopping")

args = parser.parse_args()
lr = float(args.learning_rate)
batch_size = int(args.batch_size)
model_name = str(args.model_name)
use_wandb = bool(args.wandb)
with_early_stopping = bool(args.early_stopping)


torch.cuda.empty_cache()


def get_data_loaders(mask_size):
    '''
    Returns the dataloader for training, validation and testing
    '''
    suffix = '_tiled' if tiled else ''
    train_dataset = DtdDataset('data/dtd_train{}'.format(suffix), mask_size, augmentation=None,
                               # TODO get_training_augmentation(),
                               preprocessing=get_preprocessing())
    valid_dataset = DtdDataset('data/dtd_val{}'.format(suffix), mask_size, augmentation=None,
                               # TODO get_validation_augmentation(),
                               preprocessing=get_preprocessing())

    # only use subset for testing
    # torch.manual_seed(1)
    # indices = torch.randperm(len(train_dataset)).tolist()
    # train_dataset = torch.utils.data.Subset(train_dataset, indices[:500])
    # indices = torch.randperm(len(valid_dataset)).tolist()
    # valid_dataset = torch.utils.data.Subset(valid_dataset, indices[:200])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                              multiprocessing_context=torch.multiprocessing.get_context('spawn'))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=1,
                              multiprocessing_context=torch.multiprocessing.get_context('spawn'))
    return train_loader, valid_loader


def get_model():
    '''
    Returns the model and the mask size
    '''
    in_channels = 3 if tiled else 1
    if model_name == 'simple_fcn':
        model = SimpleFullyCnn(in_channels=in_channels, out_channels=num_classes).to(device)
        mask_size = model.get_mask_size()

    elif model_name == 'simple_u-net':
        model = SimpleUnet(in_channels=in_channels, out_channels=num_classes).to(device)
        mask_size = (128, 128)

    elif model_name == 'pretrained_u-net':
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name='resnet18',
            encoder_weights='imagenet',
            classes=num_classes,
            in_channels=in_channels,
            decoder_use_batchnorm=True,
            activation='softmax2d'
        )
        mask_size = (128, 128)
    else:
        raise NotImplementedError("no such model: {}".format(model_name))

    print("Model has {} parameters".format(sum(p.numel() for p in model.parameters())))
    print(model)

    return model, mask_size


def get_loss(mask_size):
    if loss == 'bce-with-logits':
        return WeightedBCEWithLogitsLoss(mask_size)
    elif loss == 'dice':
        return DiceLoss()
    elif loss == 'cross-entropy':
        l = torch.nn.CrossEntropyLoss()
        l.__name__ = 'Cross Entropy Loss'
        return l
    else:
        NotImplementedError("Unkonwn loss: {}".format(loss))


def main():
    '''
    Execute training
    '''
    import wandb
    if use_wandb:
        wandb.init(project="compvis_dtd_{}".format(model_name))
        wandb.config.update({"Model": model_name,
                             "Learning Rate": lr,
                             "Batch Size": batch_size
                             })
    else:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()

    model, mask_size = get_model()
    train_loader, valid_loader = get_data_loaders(mask_size)

    loss_ = get_loss(mask_size)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    metrics = [
        Accuracy(),
        AccuracyT1(),
        Fscore(),
        Recall(),
        Precision(),
    ]

    train_epoch = TrainEpoch(model, loss=loss_, metrics=metrics, optimizer=optimizer, device=device, verbose=True)
    valid_epoch = ValidEpoch(model, loss=loss_, metrics=metrics, device=device, verbose=True)

    best_loss = 9999
    count_not_improved = 0
    for i in range(max_num_epoch):

        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        if best_loss > valid_logs[loss.__name__] + 0.00005:
            best_loss = valid_logs[loss.__name__]
            save_model(i, loss, model, valid_logs)
            count_not_improved = 0
        else:
            count_not_improved += 1

        if i % 10 == 0:
            save_model(i, loss, model, valid_logs)

        if use_wandb:
            logs = {"epoch": i, "train loss": train_logs[loss_.__name__], "valid loss": valid_logs[loss_.__name__]}
            for m in metrics:
                m_name = m.__name__
                logs["{} train".format(m_name)] = train_logs[m_name]
                logs["{} valid".format(m_name)] = valid_logs[m_name]
            wandb.log(logs)
        else:
            writer.add_scalar('{}/Loss/train'.format(model_name), train_logs[loss_.__name__], i)
            writer.add_scalar('{}/Loss/valid'.format(model_name), valid_logs[loss_.__name__], i)

            for m in metrics:
                m_name = m.__name__
                writer.add_scalar('{}/{}/train'.format(model_name, m_name), train_logs[m_name], i)
                writer.add_scalar('{}({}/valid'.format(model_name, m_name), valid_logs[m_name], i)

        # Early stopping
        if with_early_stopping and count_not_improved > 3:
            print("Early stopping!")
            break


def save_model(i, loss, model, valid_logs):
    if not os.path.exists('trained_models'):
        os.mkdir('trained_models')
    if use_wandb:
        torch.save(model, 'trained_models/{}-{}-{}.pth'.format(wandb.run.name, valid_logs[loss.__name__], i))
    else:
        torch.save(model, 'trained_models/{}-{}-{}.pth'.format(model_name, valid_logs[loss.__name__], i))
    print("Model saved")

if __name__ == '__main__':
    main()


# Not tiled
# Simple FCN: --learning_rate 0.001 --batch_size 1 --model_name simple_fcn --loss cross-entropy
# Simple U-Net 2: --learning_rate 0.001 --batch_size 1 --model_name simple_u-net --loss cross-entropy

# tiled
# Simple FCN: --learning_rate 0.0001 --batch_size 1 --model_name simple_fcn --loss cross-entropy
# Simple U-Net 2: --learning_rate 0.0001 --batch_size 1 --model_name simple_u-net --loss cross-entropy
# Ext U-Net 2: --learning_rate 0.0001 --batch_size 1 --model_name pretrained_u-net --loss cross-entropy