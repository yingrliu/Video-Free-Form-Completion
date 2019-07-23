"""
Author: Yingru Liu
Implementation of trainer to train and eval the model.
"""
from ImgModels.Ops.Saver import Saver
from Toolkit.VoxCelebData import ImgData as VoxData
from Toolkit.DavisData import ImgData as DavisData
from copy import deepcopy
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import torch
import os
import warnings
import torch.nn as nn
import numpy as np

class _Trainer():
    def __init__(self, args, **kwargs):
        args = _checkargs(args)
        self.netG = args.netG
        self.netD = args.netD
        if torch.cuda.is_available():
            self.netG = self.netG.cuda()
            self.netD = self.netD.cuda()
            if torch.cuda.device_count() > 1 and args.parallel:
                self.netG = nn.DataParallel(self.netG)
                self.netD = nn.DataParallel(self.netD)
        #
        self.args = args
        # Define Saver and Tensorboard Writer for training phrase only.
        if hasattr(args, 'mode') and args.mode == 'train':
            self.saver = Saver(args)
            self.saver.save_experiment_config()
            self.writer_logdir = os.path.join(self.saver.experiment_dir, 'tensorboard')
            #
            if args.dataset.lower() =='davis':
                args.resize = (384, 512)
            print(args.resize)
            self.trainSet, self.testSet = _dataloader(args.split_ratio, args.batch_size, args.shuffle,
                                                          args.num_workers, args.pin_memory, args.resize, args.dataset)
            # Define Optimizer
            if args.Optim.lower() == 'sgd':
                self.OptimG = torch.optim.SGD(self.netG.parameters(), lr=args.lrG, weight_decay=args.weight_decay,
                                              momentum=args.momentum, nesterov=args.nesterov)
                self.OptimD = torch.optim.SGD(self.netD.parameters(), lr=args.lrD, weight_decay=args.weight_decay,
                                              momentum=args.momentum, nesterov=args.nesterov)
            elif args.Optim.lower() == 'adam':
                self.OptimG = torch.optim.Adam(self.netG.parameters(), lr=args.lrG)
                self.OptimD = torch.optim.Adam(self.netD.parameters(), lr=args.lrD)
            elif args.Optim.lower() == 'rmsprop':
                self.OptimG = torch.optim.RMSprop(self.netG.parameters(), lr=args.lrG, weight_decay=args.weight_decay,
                                             momentum=args.momentum)
                self.OptimD = torch.optim.RMSprop(self.netD.parameters(), lr=args.lrD, weight_decay=args.weight_decay,
                                             momentum=args.momentum)
            else:
                raise ValueError("This project only supports SGD/Adam/RMSprop.")
            #
            self.lossG, self.lossD = 0., 0.
            # Setup LR scheduler.
            if args.LRscheduler.lower() == 'steplr':
                self.LRschedulerG = torch.optim.lr_scheduler.StepLR(optimizer=self.OptimG, step_size=args.step_size,
                                                                    gamma=args.gamma)
                self.LRschedulerD = torch.optim.lr_scheduler.StepLR(optimizer=self.OptimD, step_size=args.step_size,
                                                                    gamma=args.gamma)
            elif args.LRscheduler.lower() == 'multisteplr':
                self.LRschedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.OptimG, milestones=args.milestones,
                                                                    gamma=args.gamma)
                self.LRschedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.OptimD, milestones=args.milestones,
                                                                    gamma=args.gamma)
            elif args.LRscheduler.lower() == 'exponentiallr':
                self.LRschedulerG = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.OptimG, gamma=args.gamma)
                self.LRschedulerD = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.OptimD, gamma=args.gamma)
            elif args.LRscheduler.lower() == 'cosineAnnealinglr':
                self.LRschedulerG = torch.optim.lr_scheduler.CosineAnnealingLr(optimizer=self.OptimG, T_max=args.T_max,
                                                                               eta_min=args.eta_min)
                self.LRschedulerD = torch.optim.lr_scheduler.CosineAnnealingLr(optimizer=self.OptimD, T_max=args.T_max,
                                                                               eta_min=args.eta_min)
            elif args.LRscheduler.lower() == 'reducelronplateau':
                self.LRschedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.OptimG, mode=args.mode,
                                                                               factor=args.factor, patience=args.patience,
                                                                               verbose=args.verbose,
                                                                               threshold=args.threshold,
                                                                               threshold_mode=args.threshold_mode,
                                                                               cooldown=args.cooldown, min_lr=args.min_lr)
                self.LRschedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.OptimD, mode=args.mode,
                                                                               factor=args.factor, patience=args.patience,
                                                                               verbose=args.verbose,
                                                                               threshold=args.threshold,
                                                                               threshold_mode=args.threshold_mode,
                                                                               cooldown=args.cooldown, min_lr=args.min_lr)
            else:
                raise ValueError("This project only supports the following LR scheduler: "
                                 "StepLR/MultiStepLR/CosineAnnealingLR/ReduceLROnPlateau.")
            #
            self.cur_epoch = 0
        return

    def saveModels(self):
        checkPath = self.saver.experiment_dir
        #
        state_dict_G = self.netG.module.state_dict() if isinstance(self.netG, nn.DataParallel)\
            else self.netG.state_dict()
        state_dict_D = self.netD.module.state_dict() if isinstance(self.netD, nn.DataParallel) \
            else self.netD.state_dict()
        torch.save({
            'epoch': self.cur_epoch,
            'Generator': state_dict_G,
            'Discriminator': state_dict_D,
            'optimG': self.OptimG.state_dict(),
            'optimD': self.OptimD.state_dict(),
        }, os.path.join(checkPath, 'CheckPoint.pth-{:04d}.tar'.format(self.cur_epoch)))

    def loadG(self, PATH=None):
        """
        use to load the generator only.
        :param PATH:
        :return:
        """
        path = PATH if PATH else os.path.join(self.saver.experiment_dir, 'CheckPoint.pth-{:04d}.tar'.format(0))
        print("-- Load Generator from %s." % path)
        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location='cpu')
        self.netG.load_state_dict(checkpoint['Generator'])
        return

    def loadModels(self, PATH):
        path = PATH if PATH else os.path.join(self.saver.experiment_dir, 'CheckPoint.pth.tar')
        print("-- Load Model from %s." % path)
        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location='cpu')
        #
        self.netG.load_state_dict(checkpoint['Generator'])
        self.netD.load_state_dict(checkpoint['Discriminator'])
        self.OptimG.load_state_dict(checkpoint['optimG'])
        self.OptimD.load_state_dict(checkpoint['optimD'])
        self.cur_epoch = checkpoint['epoch']
        return

    def train(self):
        train_loss = 0.0
        self.netG.train()
        self.netD.train()
        # check if load model.
        if self.args.resume is not None:
            self.loadModels(self.args.resume)
        #
        for epoch in range(self.args.max_epoches):
            self.cur_epoch += 1
            records = self._train_epoch(self.cur_epoch)
            _, Imgs = self._evaluate_epoch(self.args.visualize)
            if epoch % 5 == 0:
                self.saveModels()
            if self.args.visualize:
                self._writeEvents(epoch=self.cur_epoch, dict_scalar=records, Imgs=Imgs)
            # # adjust lr.
            # self.LRschedulerD.step()
            # self.LRschedulerG.step()
        return

    def valuation(self):
        return

    def _train_epoch(self, epoch):
        return None

    def _evaluate_epoch(self, visualize=False):
        return None, None

    def _writeEvents(self, epoch, dict_scalar=None, Imgs=None):
        """
        write training curves into tensorboard.
        :param epoch:
        :param dict_scalar: dictionary of scalar record.
        :param Imgs: The batch of images.
        :return:
        """
        writer = SummaryWriter(self.writer_logdir)
        if dict_scalar is not None:
            for key, value in dict_scalar.items():
                writer.add_scalar(tag=key, scalar_value=value, global_step=epoch)
        if Imgs is not None:
            grid_image = make_grid(Imgs, 5, normalize=True)
            writer.add_image('Image', grid_image, epoch)
        writer.close()
        return



def _dataloader(split_ratio=0.8, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, resize=(512, 512),
                dataset='voxceleb1'):
    """
    generate the dataloader for training/testing.
    :param batch_size:
    :param shuffle:
    :param num_workers:
    :param pin_memory:
    :return:
    """
    if dataset.lower() == 'voxceleb1':
        trainSet = VoxData(resize=resize)
        testSet = deepcopy(trainSet)
        #
        split = int(len(trainSet.files) * split_ratio)
        trainSet.files = trainSet.files[0:split]
        testSet.files = testSet.files[split:]
        #
        trainSet.sketches = trainSet.sketches[0:split]
        testSet.sketches = testSet.sketches[split:]
        #
        trainSet.colors = trainSet.colors[0:split]
        testSet.colors = testSet.colors[split:]
    elif dataset.lower() =='davis':
        trainSet = DavisData(resize=resize, train=True)
        testSet = DavisData(resize=resize, train=False)
    else:
        raise ValueError("dataset should be voxceleb1/davis.")
    # create dataloader.
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
    trainSet = DataLoader(trainSet, batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory,
                          worker_init_fn=worker_init_fn)
    testSet = DataLoader(testSet, batch_size, num_workers=num_workers, shuffle=False, pin_memory=pin_memory,
                         worker_init_fn=worker_init_fn)
    return trainSet, testSet

def _checkargs(args):
    """
    check the item of the argumenets.
    :param args:
    :return:
    """
    # load defaul setting to some arguments.
    ################################################################################################################
    # Training setting I.
    if not hasattr(args, 'dataset'):
        warnings.warn("No dataset is specified. Use default value (DAVIS).")
        args.dataset = 'DAVIS'
    if not hasattr(args, 'split_ratio'):
        warnings.warn("No split ratio is specified for the dataset. Use default value (0.8).")
        args.split_ratio = 0.8
    if not hasattr(args, 'batch_size'):
        warnings.warn("No batch size is specified for the dataset. Use default value (24).")
        args.batch_size = 24
    if not hasattr(args, 'shuffle'):
        warnings.warn("No shuffle is specified for the training dataset. Use default value (True).")
        args.shuffle = True
    if not hasattr(args, 'num_workers'):
        warnings.warn("No num_workers is specified for the training dataset. Use default value (4).")
        args.num_workers = 0
    if not hasattr(args, 'pin_memory'):
        warnings.warn("No pin_memory is specified for the training dataset. Use default value (True).")
        args.pin_memory = True
    if not hasattr(args, 'resize'):
        warnings.warn("No resize is specified for the training dataset. Use default value ((512, 512)).")
        args.resize = (512, 512)
    ################################################################################################################
    # Training setting II.
    if not hasattr(args, 'max_epoches'):
        warnings.warn("No max_epoches is specified for the training process. Use default value (50).")
        args.max_epoches = 50
    if not hasattr(args, 'tolerance'):
        warnings.warn("No tolerance is specified for the training process. Use default value (5).")
        args.tolerance = 5
    if not hasattr(args, 'parallel'):
        warnings.warn("No parallel is specified for data parallel in multiple gpu. Use default value (True).")
        args.parallel = True
    if not hasattr(args, 'visualize'):
        warnings.warn("No visualize is specified for the training process. Use default value (False).")
        args.visualize = False
    if not hasattr(args, 'steps_dis'):
        warnings.warn("No steps_dis is specified for the number of update step in each iteration for discriminator. Use default value (1).")
        args.steps_dis = 1
    if not hasattr(args, 'resume'):
        warnings.warn("No resume is specified to load the pre-trained models. Use default value (None).")
        args.resume = None
    ################################################################################################################
    # Optim arguments.
    if not hasattr(args, 'Optim'):
        warnings.warn("No Optimizier is specified. Use SGD as default.")
        args.Optim = 'SGD'
    if args.Optim.lower() in ['sgd', 'rmsprop']:
        if not hasattr(args, 'weight_decay'):
            warnings.warn("No weight_decay is specified for SGD/RMSprop. Use default value (0.0).")
            args.weight_decay = 0.
        if not hasattr(args, 'momentum'):
            warnings.warn("No momentum is specified for SGD/RMSprop. Use default value (0.0).")
            args.momentum = 0.
        if args.Optim.lower() == 'sgd' and not hasattr(args, 'nesterov'):
            warnings.warn("No nesterov is specified for SGD. Use default value (False).")
            args.nesterov = False
    ################################################################################################################
    # LR scheduler arguments.
    if not hasattr(args, 'LRscheduler'):
        warnings.warn("No LR scheduler is specified. Use StepLR w. gamma=1 (constant LR) as default.")
        args.LRscheduler = 'StepLR'
    if args.LRscheduler.lower() == 'steplr':
        if not hasattr(args, 'step_size'):
            warnings.warn("No step_size is specified for StepLR. Use default value (100).")
            args.step_size = 100
        if not hasattr(args, 'gamma'):
            warnings.warn(
                "No gamma is specified for StepLR. Use default value (1.), which corresponds to constant learning rate.")
            args.gamma = 1.
    elif args.LRscheduler.lower() == 'multisteplr':
        if not hasattr(args, 'milestones'):
            warnings.warn("No milestones is specified for MultiStepLR. Use default value ([100])")
            args.milestones = [100]
        if not hasattr(args, 'gamma'):
            warnings.warn(
                "No gamma is specified for MultiStepLR. Use default value (1.), which corresponds to constant learning rate.")
            args.gamma = 1.
    elif args.LRscheduler.lower() == 'exponentiallr':
        if not hasattr(args, 'gamma'):
            warnings.warn(
                "No gamma is specified for ExponentialLR. Use default value (1.), which corresponds to constant learning rate.")
            args.gamma = 1.
    elif args.LRscheduler.lower() == 'cosineAnnealinglr':
        if not hasattr(args, 'eta_min'):
            warnings.warn("No eta_min is specified for CosineAnnealingLR. Use default value (0.).")
            args.eta_min = 1.
        if not hasattr(args, 'T_max'):
            warnings.warn("No T_max is specified for CosineAnnealingLR. Use default value (100).")
            args.T_max = 100
    elif args.LRscheduler.lower() == 'reducelronplateau':
        if not hasattr(args, 'mode'):
            warnings.warn("No mode is specified for ReduceLROnPlateau. Use default value ('min').")
            args.mode = 'min'
        if not hasattr(args, 'factor'):
            warnings.warn("No factor is specified for ReduceLROnPlateau. Use default value (.1).")
            args.factor = 0.1
        if not hasattr(args, 'patience'):
            warnings.warn("No patience is specified for ReduceLROnPlateau. Use default value (10).")
            args.patience = 10
        if not hasattr(args, 'verbose'):
            warnings.warn("No verbose is specified for ReduceLROnPlateau. Use default value (False).")
            args.verbose = False
        if not hasattr(args, 'threshold'):
            warnings.warn("No threshold is specified for ReduceLROnPlateau. Use default value (1e-4).")
            args.threshold = 1e-4
        if not hasattr(args, 'threshold_mode'):
            warnings.warn("No threshold_mode is specified for ReduceLROnPlateau. Use default value ('rel').")
            args.threshold_mode = 'rel'
        if not hasattr(args, 'cooldown'):
            warnings.warn("No cooldown is specified for ReduceLROnPlateau. Use default value (0).")
            args.cooldown = 0
        if not hasattr(args, 'min_lr '):
            warnings.warn("No min_lr is specified for ReduceLROnPlateau. Use default value (0).")
            args.min_lr = 0
    return args