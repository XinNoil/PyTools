import os
from .io import check_dir, save_json
from .py import list_con, for_print

def repeat_args(cmds, for_arg_lists):
    return list_con([['%s %s'%(cmd, for_arg_list) for for_arg_list in for_arg_lists] for cmd in cmds])

def get_cmds(cmds, for_args, is_print=False):
    for for_arg in for_args:
        for_arg_lists = []
        for for_arg_values in zip(*for_arg.values()):
            for_arg_list = []
            for for_arg_key, for_arg_value in zip(for_arg.keys(), for_arg_values):
                for_arg_list.append(f'{for_arg_key} {for_arg_value}')
            for_arg_lists.append(' '.join(for_arg_list))
        cmds = repeat_args(cmds, for_arg_lists)
    cmds = ['%s -i %d'%(cmd, i+1) for i,cmd in zip(range(len(cmds)), cmds)]
    if is_print:
        for_print(cmds)
    return cmds