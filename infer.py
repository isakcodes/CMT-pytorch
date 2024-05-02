import os
import glob
import argparse
import torch
from utils import logger
from utils.hparams import HParams
from utils.utils import get_optimizer
from dataset import get_loader
from model import ChordConditionedMelodyTransformer as CMT
from inferrer import CMTInferencer

# Hyperparameters - using argparse and parameter module
parser = argparse.ArgumentParser()
parser.add_argument('--asset_root', type=str, help='asset root (where idxNNN folders are)')
parser.add_argument('--idx', type=int, help='experiment number', default=0)
parser.add_argument('--gpu_index', '-g', type=int, default="0", help='GPU index')
parser.add_argument('--ngpu', type=int, default=4, help='0 = CPU.')
parser.add_argument('--restore_epoch', type=int, default=-1)
parser.add_argument('--load_rhythm', dest='load_rhythm', action='store_true')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:%d" % args.gpu_index if use_cuda else "cpu")

hparam_file = os.path.join(os.getcwd(), "hparams.yaml")

config = HParams.load(hparam_file)
data_config = config.data_io
model_config = config.model
exp_config = config.experiment

# Configuration
asset_root = args.asset_root
asset_path = os.path.join(asset_root, 'idx%03d' % args.idx)
logger.add_filehandler(os.path.join(asset_path, "log_inference.txt"))

# Get data loader for inference
logger.info("Getting data loader for inference")
test_loader = get_loader(data_config, mode='test')

# Build model
logger.info("Building model")
model = CMT(**model_config)

if args.ngpu > 1:
    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
model.to(device)

# Get inferencer
inferencer = CMTInferencer(asset_path, model, device, config)
# load model if exists
inferencer.load_model(args.restore_epoch, args.load_rhythm)

inferencer.model.eval()
loader = inferencer.test_loader

# Start inference
logger.info("Starting inference")
inferencer.infer(args.restore_epoch, args.load_rhythm)
