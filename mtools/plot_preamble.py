# -*- coding: UTF-8 -*-

## Jupyter Template
"""
## change dir
import os
import yaml
if os.path.exists('../Configs/Local/local.yaml'):
    data = yaml.load(open('../Configs/Local/local.yaml', 'r'), Loader=yaml.FullLoader)
    os.chdir(data['proj_dir'])

## import plot tools
from mtools.plot_preamble import *
from mtools import list_ind, unique, sort_df, select_df, remove_row
%matplotlib inline
"""

import os
import os.path as osp

import sys
import ipdb as pdb
import numpy as np
np.set_printoptions(precision=4, suppress=True, formatter={'float_kind':'{:f}'.format})

import pandas as pd
pd.set_option('display.float_format','{:.4f}'.format)
pd.set_option('display.width', 500)

import seaborn as sns
sns.set(**{'style':'whitegrid', 'font':'Times New Roman'})
sns.set_style("whitegrid", {"axes.edgecolor": "0"})

import matplotlib.pyplot as plt
import matplotlib

from IPython.display import display
from mtools import list_ind

matplotlib.rcParams['font.family'] = ['Times New Roman']
matplotlib.rcParams['font.style'] = 'normal'
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.unicode_minus'] = False

def setfontsize(fontsize):
    matplotlib.rcParams['axes.titlesize'] = fontsize
    matplotlib.rcParams['axes.labelsize'] = fontsize
    matplotlib.rcParams['figure.titlesize'] = fontsize
    matplotlib.rcParams['xtick.labelsize'] = fontsize
    matplotlib.rcParams['ytick.labelsize'] = fontsize
    matplotlib.rcParams['legend.fontsize'] = fontsize
    matplotlib.rcParams['legend.title_fontsize'] = fontsize

setfontsize(18)
colors = sns.color_palette()
fontsize = 18
fontdict={"family":"Times New Roman", 'size':fontsize, 'fontweight':'bold'}

def save_fig(g, fig_name):
    g.figure.savefig(f'{fig_name}.png', bbox_inches='tight')
    g.figure.savefig(f'{fig_name}.pdf', bbox_inches='tight', transparent=True, pad_inches=0)

def set_g(g, fontsize=18, xlabel='', ylabel='', title='', is_text=False, text_fmt='%.1f', bar_label_fontsize=None, fontdict={'fontweight':'bold'}, **kwargs):
    fontdict['fontsize'] = fontsize
    g.set_xlabel(xlabel, fontdict=fontdict)
    g.set_ylabel(ylabel, fontdict=fontdict)
    g.set_title(title, fontdict=fontdict)
    g.set(**kwargs)
    if is_text:
        for i, container in enumerate(g.containers):
            g.bar_label(container, fmt=text_fmt, fontsize=fontsize if bar_label_fontsize is None else bar_label_fontsize)
    plt.legend()
    g.get_legend().set_title('')

def set_gs(g, fontsize=18, xlabel='', ylabel='', title='', fontdict={'fontweight':'bold'}, **kwargs):
    fontdict['fontsize'] = fontsize
    g.set_xlabels(xlabel, fontdict=fontdict)
    g.set_ylabels(ylabel, fontdict=fontdict)
    g.set_titles(title, fontdict=fontdict)
    g.set(**kwargs)
    plt.legend()
    g.legend.set_title('')

def set_hatch(g, hidx=[], hatch_num=1, hatchs = ['/', '\\', '|', '-', '+', 'x', '.', 'o', 'O', '*', '']):
    _hatchs = list_ind(hatchs, hidx) if len(hidx)>0 else hatchs
    for i, container in enumerate(g.containers):
        for _bar in container:
            _bar.set_hatch(_hatchs[i]*hatch_num)