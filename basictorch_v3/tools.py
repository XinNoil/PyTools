import os, time, torch
import mtools.monkey as mk
from mtools import read_file, write_file
from torch.utils.data import DataLoader, Dataset
from hydra.core.hydra_config import HydraConfig
from time import sleep

def item_losses(losses):
    for loss in losses:
        losses[loss] = losses[loss].item()
    return losses

def n2t(num, tensortype=torch.FloatTensor, device=None, **kwargs):
    return tensortype(num, **kwargs).to(mk.get_current_device() if device is None else device)

def t2n(tensor):
    return tensor.detach().cpu().numpy()

def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_model(model):
    for p in model.parameters():
        p.requires_grad = True

def data_to_device(batch_data, device=None):
    device = mk.get_current_device() if device is None else device
    if isinstance(batch_data, torch.Tensor):
        return batch_data.to(device)
    else:
        return tuple(data_to_device(item, device) for item in batch_data)

def get_test_dataset(test_dataset, batch_size):
    if issubclass(type(test_dataset), DataLoader):
        return test_dataset.dataset, test_dataset
    elif issubclass(type(test_dataset), Dataset):
        return test_dataset, DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        raise TypeError(f"Unexpected test_dataset type :{type(test_dataset)}")

def get_dataset_property(dataset, name):
    if hasattr(dataset, name):
        return getattr(dataset, name)
    else:
        return get_dataset_property(dataset.dataset, name)

class Obj(object):
    def __init__(self, obj_dict) -> None:
        self.__dict__ = obj_dict
    
    def get(self, __name, __default=None):
        return getattr(self, __name, __default)
    
    def keys(self):
        return self.__dict__.keys()

def dictConfig_to_dict(cfg):
    return Obj(dict(zip(cfg.keys(), cfg.values())))

def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def hydra_log_info(log):
    log.info(f"Override Params: {HydraConfig.get().overrides.task}")
    log.info(f"Output Dir: {HydraConfig.get().runtime.output_dir}")

def is_cmd(cmd):
    return (not cmd.startswith('#')) and len(cmd)

def join_cmds(cmds):
    cmds = [_.strip().strip('\\') for _ in cmds]
    cmds = list(filter(is_cmd, cmds))
    new_cmds = []
    new_cmd = None
    for cmd in cmds:
        if cmd.startswith('python'):
            if new_cmd is not None:
                new_cmds.append(new_cmd)
            new_cmd = cmd
        else:
            if new_cmd is None:
                raise Exception('first line has to startwiths python')
            new_cmd += cmd
    if new_cmd is not None:
        new_cmds.append(new_cmd)
    return new_cmds

def load_cmds(filename):
    cmds = read_file(filename)
    cmds = join_cmds(cmds)
    return cmds

def get_dev(dev_use_nums):
    return dev_use_nums.index(min(dev_use_nums))

def long_time_task(args, i, seed, cmd=None, pool=None, dev=None):
    prefix = '%d/%d-%s/%d'%(i, args.cmd_num-1, args.seed.index(seed)+1 if seed in args.seed else 0, len(args.seed))
    _cmd = cmd
    param = ''
    cmd = cmd.replace('python ', 'python -W ignore ')
    cmd = cmd.replace('python3 ', 'python3 -W ignore ')
    if dev is not None:
        param += ' Trainer.device=cuda:%d' % (dev)
    
    if pool is not None:
        param += ' +Trainer.task_i=%s +Trainer.task_p=%d +Trainer.process_bar=epoch'%(prefix, pool)
    if args.append is not None:
        param += ' %s'%args.append
    
    if seed is not None:
        param += ' seed=%d'%seed
    if args.log and pool is not None:
        param += ' > %s_%s.log'%(os.path.join(args.path.replace('.sh',''), 'run%d'%(i)), args.timestamp)
        if seed is not None:
            param = param.replace('.log', '_seed%d.log'%(seed))
    if '#' in cmd:
        cmd = cmd.replace('#', ' %s #'%param, 1)
    else:
        cmd += param
    
    status = 0
    
    if args.test==0 and pool is not None:
        try:
            now_time = time.strftime('%y/%m/%d-%H:%M:%S', time.localtime(time.time()))
            print('\nRun %s on cuda:%d started at %s'%(prefix, dev, now_time))
            status = os.system(cmd)
            now_time = time.strftime('%y/%m/%d-%H:%M:%S', time.localtime(time.time()))
            print('\nRun %s on cuda:%d done at %s'%(prefix, dev, now_time))
        except:
            print('\nRun %s Error'%prefix)
            status = 1
    else:
        print('\nRun %s: %s' % (prefix, cmd[:100]))
    return i, seed, _cmd, pool, dev, status

def get_cmds_list(args, cmds):
    cmds_list = []
    for seed in args.seed:
        for i,cmd in enumerate(cmds):
            long_time_task(args, i, seed, cmd)
            cmds_list.append((i, seed, cmd))
    return cmds_list

def run_loop(p, args, cmds_list, callback):
    for i, seed, cmd in cmds_list:
        pool, dev = get_pool_dev(args)
        p.apply_async(long_time_task, args=(args, i, seed, cmd, pool, dev), callback=callback)
    
def get_pool_dev(args, try_num=0):
    pool = args.pool_use_nums.index(min(args.pool_use_nums))
    args.pool_use_nums[pool] += 1
    if args.dev[0] != -1:
        dev = get_dev(args.dev_use_nums)
        if args.dev_use_nums[dev] == args.max_dev_num:
            if try_num<5:
                sleep(0.1)
                return get_pool_dev(args, try_num+1)
        args.dev_use_nums[dev] += 1
    else:
        dev = None
    return pool, dev

