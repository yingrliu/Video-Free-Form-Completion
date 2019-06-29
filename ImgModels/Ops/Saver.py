import os
import shutil
import torch
import yaml
import glob

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join(args.abs_path, 'run', args.dataset, args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_experiment_config(self):
        config = vars(self.args)
        config.pop('netG')
        config.pop('netD')
        config.pop('abs_path')
        with open(os.path.join(self.experiment_dir, 'parameters.yaml'), "w") as yaml_file:
            yaml.dump(config, yaml_file)
        yaml_file.close()
        return