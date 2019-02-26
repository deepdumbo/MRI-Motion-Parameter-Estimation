import os
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Run experiment to train several models specified in a list of config files.')
parser.add_argument('experiment',help = 'Path to .txt experiment file.')

args = parser.parse_args()
experiment_path = args.experiment

experiment_file = open(experiment_path,'r')
experiments = experiment_file.read().splitlines()

config_path = '/data/vision/polina/scratch/nmsingh/dev/Motion-Experiments/configs'

for experiment in experiments:
    this_config = os.path.join(config_path,experiment)
    subprocess.Popen(['srun --gres=gpu:titan -t 0 python train.py ' + this_config],shell=True)
