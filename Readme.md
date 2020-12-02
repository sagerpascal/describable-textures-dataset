# Semantic Segmentation on Describable Textures Dataset
*This repository was created as a homework assignment during my Msc. studies in Data Science.*

The data set consists of images with different structures which have to be classified. However, 
instead of a simple label a mask must be predicted (semantic segmentation).
For the first network, only conv layers may be used. For the second network the choice is 
of the network architecture is free. I decided to use a U-Net, which was developed by the 
University of Freiburg, Germany.

### Setup
Create the conda environment and activate it:
```shell script
conda env create -f environment.yml
conda activate describable-textures-dataset
```

Download the images:
```shell script
python dtd_loader_color_patches.py
```

### Mask Distribution
The following plot shows the distribution of the mask:

<img src="doc/img/mask_distribution.png" alt="mask_distribution" style="zoom:80%;" />

This distribution was considered with a weighted loss function.




### Train a model
```shell script
python train [OPTIONS]
```

Arguments:
- `--learning_rate`: The learning rate
- `--batch_size`: The batch size
- `--model_name`: Name of the model, one of `'simple_fcn'`, `'simple_u-net'` or `'pretrained_u-net'`
- `--wandb`: Set to `True` if you want to use [wandb.ai](wandb.ai), default is using Tensorboard

##### Run Sweep
With wandb, it is also possible to run sweeps. First, define the `model_name` in the
File `sweep.yaml` and then execute:

```
wandb sweep sweep.yaml
wandb agent your-sweep-id
```

### Run a pretrained model
TODO

### Results
