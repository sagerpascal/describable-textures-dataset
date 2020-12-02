import argparse
import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import DtdDataset, get_training_augmentation, get_validation_augmentation, get_preprocessing
from epoch import TrainEpoch, ValidEpoch
from metrics import Accuracy, Fscore, Recall, Precision, AccuracyT1
from models import SimpleFullyCnn, SimpleUNet
import yaml

# Read Config
conf = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
num_classes = conf['num_classes']
device = conf['device']
max_num_epoch = conf['max_num_epoch']

# Read parameters from command line
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate")
parser.add_argument("--batch_size")
parser.add_argument("--model_name")
parser.add_argument("--wandb")

args = parser.parse_args()
lr = float(args.learning_rate)  # 0.000005
batch_size = int(args.batch_size)  # 8
model_name = str(args.model_name)
use_wandb = bool(args.wandb)

torch.cuda.empty_cache()


def get_data_loaders(mask_size):
    '''
    Returns the dataloader for training, validation and testing
    '''
    train_dataset = DtdDataset('data/dtd_train_tiled', mask_size, augmentation=get_training_augmentation(),
                               preprocessing=get_preprocessing())
    valid_dataset = DtdDataset('data/dtd_val_tiled', mask_size, augmentation=get_validation_augmentation(),
                               preprocessing=get_preprocessing())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, multiprocessing_context=torch.multiprocessing.get_context('spawn'))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=1, multiprocessing_context=torch.multiprocessing.get_context('spawn'))
    return train_loader, valid_loader


class MyWeightedLoss(torch.nn.BCEWithLogitsLoss):
    '''
    Weighted BCEWithLogitsLoss (weighted according mask distribution)
    '''
    __name__ = 'weighted-bce-with-logits-loss'

    def __init__(self, mask_size, **kwargs):
        pos_weight = np.ones((num_classes, mask_size[0], mask_size[1]))
        weights = [0.956569204, 0.977921196, 1.408709632, 1.084427464, 1.315641729, 0.946239083, 0.952410208,
                   0.905183255, 1.038172265, 0.82662018, 0.803869166, 0.995701581, 1.018857432, 1.114780396,
                   0.905183255, 1.000248164, 0.944199775, 1.043115942, 0.954485176, 1.092540388, 0.946239083,
                   0.938134252, 1.076434142, 0.825063457, 0.914631932, 0.842516722, 1.076434142, 1.040638232,
                   1.014140499, 1.055683604, 0.973574879, 1.103548352, 1.269880277, 1.120482598, 0.954485176,
                   0.876217391, 1.087118351, 0.847405601, 0.960764683, 1.505528164, 1.929994254, 0.857355569,
                   0.895927803, 0.82662018, 0.988958681, 1.055683604]
        for i, w in enumerate(weights):
            pos_weight[i, :, :] = pos_weight[i, :, :] * w
        pos_weight = torch.FloatTensor(pos_weight)
        super().__init__(pos_weight=pos_weight, **kwargs)

    def forward(self, y_pr, y_gt):
        return super(MyWeightedLoss, self).forward(y_pr, y_gt)


def get_model():
    '''
    Returns the model and the mask size
    '''
    if model_name == 'simple_fcn':
        model = SimpleFullyCnn(in_channels=3, out_channels=num_classes).to(device)
        mask_size = model.get_mask_size()

    elif model_name == 'simple_u-net':
        model = SimpleUNet(input_channels=3, output_channels=num_classes).to(device)
        mask_size = (128, 128)

    elif model_name == 'pretrained_u-net':
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            classes=num_classes,
            in_channels=3,
            decoder_use_batchnorm=True,
            activation='softmax2d'
        )
        mask_size = (128, 128)
    else:
        raise NotImplementedError("no such model: {}".format(model_name))

    return model, mask_size


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
    loss = MyWeightedLoss(mask_size)

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
    for i in range(max_num_epoch):
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
        if count_not_improved > 5:
            print("Early stopping!")
            break


if __name__ == '__main__':
    main()
