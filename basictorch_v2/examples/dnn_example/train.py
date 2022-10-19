import numpy as np
import argparse
from basictorch_v2.trainer import get_optim, Trainer
from basictorch_v2.models import DNN
from basictorch_v2.losses import loss_funcs
from basictorch_v2.tools import set_args_config, set_device
from datahub.updatedata import get_h5_DB
from mtools import str2bool
from data.dataset import WiFiDataset, Datasets

# Parsing arguments
parser = argparse.ArgumentParser(description='paramters for training the reg models')
parser.add_argument('-c','--config',        type=str, nargs='+', default=None , help='config names')
# model arguments
parser.add_argument('-m','--model',         type=str,           default='dnn',     help='model name, value=[dnn,fit]')
parser.add_argument('-act','--activations', type=str,           default='relu')
parser.add_argument('-l','--layer_units',   type=int,           default=[256,128,32],   nargs='+', help='hidden layers units, eg: -l 256 128 32')
parser.add_argument('-d','--dropouts',      type=float,         default=[0.05, 0.0],    nargs='+')

# data arguments 
parser.add_argument('-n','--data_name',     type=str,           default='200925-D1')
parser.add_argument('-pf','--data_postfix', type=str,           default=None,       help='data postfix, such as intv60')
parser.add_argument('-v','--data_ver',      type=str,           default='h5',       help='data version, the folder name in DeepPrintData')
parser.add_argument('-fm','--feature_mode', type=str,           default='R',        help='')
parser.add_argument('-lm','--label_mode',   type=str,           default='cdns',     help='label mode, cdns')
parser.add_argument('-sf','--is_shuffle',   type=str2bool,      default=True,       help='print batch loss info')
parser.add_argument('-val','--is_validate', type=str2bool,      default=True,       help='print batch loss info')
parser.add_argument('--val_split',          type=float,         default=0.1)

# training arguments
parser.add_argument('-b','--batch_size',    type=int,           default=32,         help='batch size , default value=16')
parser.add_argument('-e','--epochs',        type=int,           default=100,        help='epochs, default value=100')
parser.add_argument('-o','--output',        type=str,           default='tmp',      help='output, the folder name of output')
parser.add_argument('-st','--start_trail',  type=int,           default=0,          help='start trail time')
parser.add_argument('-t','--trails',        type=int,           default=3,          help='trail times')
parser.add_argument('-no','--exp_no',       type=str,           default='')
parser.add_argument('-dev','--device',      type=int,           default=0,          help='device')
parser.add_argument('-pb','--print_batch',  type=str2bool,      default=True,       help='print batch loss info')
parser.add_argument('-load','--load_model', type=str2bool,      default=False)
parser.add_argument('-i','--run_i',         type=str,           default='1')

args = set_args_config(parser)
torch_device = set_device(args)
train_db = get_h5_DB(args.data_name, 'training')
test_db = get_h5_DB(args.data_name, 'testing')
test_db.set_bssids(train_db.bssids)

Ds = Datasets(args.is_shuffle, args.is_validate, args.val_split, args.batch_size, WiFiDataset(train_db), WiFiDataset(test_db))

for e in range(args.start_trail, args.trails):
    model = DNN(
        name='dnn',
        dim_x=Ds.train_dataset.get_feature_dim(), 
        dim_y=Ds.train_dataset.get_label_dim(),
        activations = args.activations,
        layer_units = args.layer_units,
        dropouts = args.dropouts
    ).to(torch_device)
    print(model)
    trainer = Trainer('%s_trainer'%args.model, args, loss_func=loss_funcs['mee'])
    trainer.outM.set_exp_no(e)
    trainer.fit(model, get_optim(model.parameters(), 'Adadelta'), Ds, args.batch_size, args.epochs)
