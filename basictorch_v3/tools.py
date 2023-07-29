import os, time, torch
from tqdm import tqdm
import mtools.monkey as mk
from mtools import read_file, save_json
from torch.utils.data import DataLoader, Dataset
from hydra.core.hydra_config import HydraConfig
from time import sleep
from git import Repo
from mtools import get_repo_status

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

def list_index(a, v):
    for i,_ in enumerate(a):
        if _ == v:
            return i
    return None

def get_dev(max_use_nums, dev_use_nums):
    if len(max_use_nums)>1:
        left_use_nums = [_max-used for _max,used in zip(max_use_nums, dev_use_nums)]
        return list_index(left_use_nums, max(left_use_nums))
    else:
        return list_index(dev_use_nums, min(dev_use_nums))

def long_time_task(args, i, seed, cmd=None, pool=None, dev=None):
    status = 0
    try:
        prefix = '%d/%d-%s/%d'%(i, args.cmd_num-1, args.seed.index(seed)+1 if seed in args.seed else 0, len(args.seed))
        _cmd = cmd
        param = ''
        cmd = cmd.replace('python ', 'python -W ignore ')
        cmd = cmd.replace('python3 ', 'python3 -W ignore ')
        if dev is not None:
            param += ' Trainer.device=cuda:%d' % (args.dev[dev])
        
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
        
        if pool is not None:
            if args.test==0:
                status = os.system(cmd)
            else:
                now_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
                pbar = tqdm(total=10, desc=f"Run {prefix} on cuda:{args.dev[dev]} started at {now_time}", position=pool, leave=False)
                for _ in range(10):
                    pbar.update()
                    sleep(0.2)
                now_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
                pbar.set_description(f"Run {prefix} on cuda:{args.dev[dev]} finished at {now_time}")
                pbar.close()
        else:
            print('Run %s: %s...' % (prefix, cmd[:120]))
    except:
        status = 1
    return i, seed, _cmd, pool, dev, status

def get_cmds_list(args, cmds):
    cmds_list = []
    for seed in args.seed:
        for i,cmd in enumerate(cmds):
            long_time_task(args, i, seed, cmd)
            cmds_list.append((i, seed, cmd))
    return cmds_list

def run_loop(p, args, cmds_list, var_lock, pool_use_nums, dev_use_nums, callback):
    for i, seed, cmd in cmds_list:
        pool, dev = get_pool_dev(args, pool_use_nums, dev_use_nums, var_lock)
        p.apply_async(long_time_task, args=(args, i, seed, cmd, pool, dev), callback=callback)
        
def get_pool_dev(args, pool_use_nums, dev_use_nums, var_lock):
    while True:
        with var_lock:
            pool = list_index(pool_use_nums, 0)
            if pool is not None:
                pool_use_nums[pool] = 1
                if args.dev[0] != -1:
                    dev = get_dev(args.num, dev_use_nums)
                    dev_use_nums[dev] += 1
                else:
                    dev = None
                break
        sleep(1)
    return pool, dev

def get_repos(cfg):
    repos = []
    for name in cfg.keys():
        repo = Repo(cfg[name])
        repo.name = name
        repos.append(repo)
    return repos

def get_git_info(cfg, filename=None):
    repos = get_repos(cfg)
    info_list = [{'name':repo.name ,'branch':repo.heads[0].name, 'commit':str(repo.heads[0].commit), 'status':get_repo_status(repo)} for repo in repos]
    if filename is not None:
        save_json(filename, info_list)
    return info_list
