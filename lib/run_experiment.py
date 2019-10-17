import os
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Run experiment to train several models specified in a list of config files.')
parser.add_argument('experiment',help = 'Path to .txt experiment file or directory containing multiple configs.')

args = parser.parse_args()
experiment_path = args.experiment

if(experiment_path[-4:]=='.txt'):
    experiment_file = open(experiment_path,'r')
    experiments = experiment_file.read().splitlines()

    config_path = '/data/vision/polina/scratch/nmsingh/dev/Motion-Experiments/configs'

    for experiment in experiments:
        this_config = os.path.join(config_path,experiment)
        subprocess.Popen(['srun -p gpu --gres=gpu:titan:1 -t 0 python train.py ' + this_config],shell=True)
else:
    for exp_config in os.listdir(experiment_path):
        this_config = os.path.join(experiment_path,exp_config)
        subprocess.Popen(['srun -p gpu --gres=gpu:titan:1 -t 0 python train.py ' + this_config + ' --suffix ' + exp_config[:-4]],shell=True)
