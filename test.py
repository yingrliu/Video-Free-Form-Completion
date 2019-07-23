from ImgModels.Ops import _Trainer
from ImgModels.SC_FEGAN import sc_fegan, sn_patch_gan, SC_FEGAN_Trainer
import argparse, os
from Toolkit.DavisData import mp_preprocessing



parser = argparse.ArgumentParser(description="")
parser.add_argument('--dataset', type=str, default='davis',
                    choices=['voxceleb1', 'davis'],
                    help=' ')
parser.add_argument('--Optim', type=str, default='adam',
                    help=' ')
parser.add_argument('--checkname', type=str, default='SC-FEGAN',
                    help=' ')
parser.add_argument('--num_workers', type=int, default=1,
                    metavar='N', help='dataloader threads')
parser.add_argument('--batch-size', type=int, default=1,
                    help=' ')
parser.add_argument('--pin_memory', type=bool, default=True,
                    help=' ')
parser.add_argument('--shuffle', type=bool, default=True,
                    help=' ')
args = parser.parse_args()
args.abs_path = os.path.dirname(os.path.abspath(__file__))

args.lrD = 5e-5
args.lrG = 5e-4
# args.batch_size = 24
args.cuda = True
args.visualize = True
#
args.lambda_per = 0.05
args.lambda_gt = 0.001
args.lambda_sn = 0.001
args.lambda_sty = 120
args.lambda_tv = 0.1
args.steps_dis = 1

if __name__ == "__main__":
    # trainer = SC_FEGAN_Trainer(args)
    # trainer.train()
    mp_preprocessing()
