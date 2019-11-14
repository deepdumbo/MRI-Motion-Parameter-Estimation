# MRI-Motion-Parameter-Estimation

Train and evaluate a neural network to prediction motion parameters from k-space data with simulated motion corruption.

To train a network, create a config file (e.g. ```model_config.ini```) with your desired specifications:
```
[DATA]
n = 64
dataset = BRAIN
corruption = ALL
corruption_extent = CONTIGUOUS
patch = False

[MODEL]
architecture = PARAMETERIZED_SINGLE_THETA
input_domain = FREQUENCY
output_domain = SINGLE_THETA
nonlinearity = relu

[TRAINING]
pretrain = False
num_epochs = 5000
```

To run, execute
```
python lib/train.py model_config.ini --train_dir $optional_output_dir --suffix $optional-identifier
```

