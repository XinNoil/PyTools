import os, sys, time, torch, math, copy, inspect, random
import numpy as np
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data import Subset,TensorDataset
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
from mtools import tojson, save_json, load_json,check_dir, colors_names, join_path, gitcommit_repos, tuple_ind
from .dataset import MDataLoader

class Base(object):
    def __init__(self, name, args, default_args={}, args_names=[], **kwargs):
        super().__init__()
        self.name = name
        self.args = args
        self.set_params(default_args, args_names, kwargs)
    
    def set_params(self, default_args, args_names, kwargs):
        self.params = kwargs
        for param in args_names:
            if (param not in self.params) and hasattr(self.args, param):
                self.params[param] = self.args.__dict__[param]
        self.params = merge_params(self.params, default_args)
        set_params(self, self.params)
    
    def apply_func(self, func=None, func_params={}):
        if func:
            func(self, **func_params)

def set_params(obj, params):
    for param in params:
        obj.__dict__[param] = copy.copy(params[param])

# train script tools
def is_args_set(arg_name, option_strings_dict):
    if '-%s'%arg_name in option_strings_dict:
        option_strings = option_strings_dict['-%s'%arg_name]
    elif '--%s'%arg_name in option_strings_dict:
        option_strings = option_strings_dict['--%s'%arg_name]
    else:
        return False
    for option_string in option_strings:
        if (option_string in sys.argv) or (option_string in sys.argv):
            return True 
    return False

def get_option_strings_dict(option_strings_list):
    option_strings_dict = {}
    for option_strings in option_strings_list:
        for option_string in option_strings:
            option_strings_dict[option_string] = option_strings
    return option_strings_dict

def _set_args_config(args, parser, path=join_path('configs', 'train_configs')):
    if hasattr(args, 'config') and (args.config is not None) and len(args.config):
        option_strings_list = [action.option_strings for action in parser._actions]
        option_strings_dict = get_option_strings_dict(option_strings_list)
        for config_name in args.config:
            config = load_json(join_path(path,'%s.json'%config_name))
            for _name in config:
                if not is_args_set(_name, option_strings_dict):
                    setattr(args, _name, config[_name])

def set_args_config(parser, path=join_path('configs', 'train_configs')):
    # args > json > default
    args = parser.parse_args()
    _set_args_config(args, parser, path)
    if hasattr(args, 'git_commit') and args.git_commit:
        gitcommit_repos(join_path('configs', 'git.json'))
    print('>> %s\n' % str(args))
    return args

torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TorchDevice(str):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def set_device(self, args):
        if (hasattr(args, 'device') and args.device) or (type(args) == str):
            device = args.device if hasattr(args, 'device') else args
            self.device = torch.device("cuda:%d"%device if torch.cuda.is_available() else "cpu")
            global torch_device
            torch_device = self.device
        print('>> device: %s\n' % str(self.device))
        return self.device
    
    def __call__(self):
        return self.device

torchDevice = TorchDevice()

def set_device(args):
    return torchDevice.set_device(torchDevice, args)

def get_arg(args, name, arg, required=False):
    if arg is None:
        if hasattr(args, name):
            arg = args.__dict__[name]
        else:
            arg = None
    if required and (arg is None):
        raise Exception('%s is required'%name)
    else:
        return arg

class OutputManger(object):
    def __init__(self, args, output=None, model_name=None, data_name=None, data_ver=None, data_postfix=None, feature_mode=None, e=None, seed=None, **kwargs):
        # required of args:
        super().__init__()
        self.output_root = 'output' # overwrite by kwargs
        self.output = get_arg(args, 'output', output, True)
        self.model_name = get_arg(args, 'model', model_name, True)
        self.data_name = get_arg(args, 'data_name', data_name, True)
        self.data_ver = get_arg(args, 'data_ver', data_ver, True)
        
        # optional:
        self.data_postfix = get_arg(args, 'data_postfix', data_postfix)
        self.feature_mode = get_arg(args, 'feature_mode', feature_mode)
        self.seed = get_arg(args, 'seed', seed)

        if e is None:
            if hasattr(args, 'exp_no'):
                if type(args.exp_no) == str:
                    self.exp_no = args.exp_no
            elif hasattr(args, 'e'):
                self.set_exp_no(args.e, seed)
            else:
                self.exp_no = ''
        else:
            self.set_exp_no(e, seed)
        
        set_params(self, kwargs)

    def set_exp_no(self, e, seed=None):
        self.exp_no = get_exp_no(self.data_postfix, e+1)

    def get_filename(self, name, file_extension='csv', out_dir=None, sub_out_dir=None, output_root=None, by_exp_no=True, model_name=None, data_name=None):
        if by_exp_no and (self.exp_no is None):
            raise Exception('exp_no is not set')
        if output_root is None:
            output_root = self.output_root
        postfix = '%s_%s' % (name, self.exp_no) if by_exp_no else name
        if data_name is None:
            data_name = self.data_name
        if out_dir is None:
            out_dir = '%s_%s' % (data_name, self.data_ver)
            out_dir = out_dir if self.feature_mode is None else '%s_%s'%(out_dir, self.feature_mode)
        if sub_out_dir is not None:
            out_dir = os.path.join(out_dir, sub_out_dir)
        model_name = self.model_name if model_name is None else model_name
        filename = '%s_%s.%s' % (model_name, postfix, file_extension)
        output_path = os.path.join(output_root, self.output, out_dir)
        return os.path.join(check_dir(output_path), filename)

def get_exp_no(data_postfix, e):
    return 'e' + str(e) if data_postfix is None else data_postfix + 'e' + str(e)

# tensor tools

def print_mem(message=''):
    print("{}".format(message, torch.cuda.memory_allocated(0)))

def n2t(num, tensortype=torch.FloatTensor, device=None, **kwargs):
    return tensortype(num, **kwargs).to(torchDevice() if device is None else device)

def t2n(tensor):
    return tensor.detach().cpu().numpy()

def stack_mean(x):
    return torch.mean(torch.stack(x), dim=0)

def one_hot_d(num, device, d, num_d):
    return torch.nn.functional.one_hot(torch.zeros(num, dtype=torch.long, device=device)+d, num_classes=num_d)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

def seed_numpy(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def data_to_device(batch_data):
    if isinstance(batch_data, torch.Tensor):
        return batch_data.to(torchDevice())
    else:
        return tuple(data_to_device(item) for item in batch_data)

# dataset tools
class MySubset(Subset):
    @property
    def tensors(self):
        return tuple(tensor[self.indices] for tensor in self.dataset.tensors)

def random_split(dataset, lengths, generator=default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()
    return [MySubset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]

def get_tensor_dataset(db, feature_mode, label_mode=None, extra_attr_list=[], extra_attr_tensortypes=[], feature_tensortype=torch.FloatTensor, label_tensortype=torch.FloatTensor):
    tensors = [n2t(db.get_feature(feature_mode), tensortype=feature_tensortype)]
    if label_mode:
        tensors.append(n2t(db.get_label(label_mode), tensortype=label_tensortype))
    for attr,tensortype in zip(extra_attr_list, extra_attr_tensortypes):        
        tensors.append(n2t(db.getattr(attr), tensortype=tensortype))
    return TensorDataset(*tensors)

# module tools

def print_foward(model):
    print(inspect.getsource(model.forward))

def get_layers(input_dim, layer_units, Layer = torch.nn.Linear, **kwargs):
    layers = torch.nn.ModuleList()
    if len(layer_units)>0:
        layers.append(Layer(input_dim, layer_units[0], **kwargs))
        for l in range(len(layer_units)-1):
            layers.append(Layer(layer_units[l], layer_units[l+1], **kwargs))
    return layers

def spectral_norm(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
        torch.nn.utils.spectral_norm(m)
    elif isinstance(m, torch.nn.LSTM):
        name_pre = 'weight'
        for i in range(m.num_layers):
            name = name_pre+'_hh_l'+str(i)
            torch.nn.utils.spectral_norm(m, name)
            name = name_pre+'_ih_l'+str(i)
            torch.nn.utils.spectral_norm(m, name)

def initialize_model(model):
    for m in model.modules():
        if issubclass(type(m), torch.nn.Linear):
            torch.nn.init.uniform_(m.weight, -0.05, 0.05)
        elif issubclass(type(m), (torch.nn.Conv1d, torch.nn.Conv2d)):
            torch.nn.init.xavier_uniform_(m.weight)
        elif hasattr(m, 'reset_parameters'):
            m.reset_parameters()

def reset_parameters(model):
    for m in model.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

def save_model(model, outM=None, postfix='', filename=None, model_name=None):
    if filename is None:
        filename = outM.get_filename('model_%s%s' % (model.name, postfix), 'pth', model_name=model_name)
    torch.save(model.state_dict(), filename)

def load_model(model, outM=None, postfix='', filename=None, model_name=None, strict=True):
    if filename is None:
        filename = outM.get_filename('model_%s%s' % (model.name, postfix), 'pth', model_name=model_name)
    print('load from:', filename)
    model.load_state_dict(torch.load(filename, map_location=torchDevice()), strict=strict)

def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_model(model):
    for p in model.parameters():
        p.requires_grad = True

def unfreeze_optimizer(optimizer):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            param.requires_grad = True

def train_bn(m):
    if m.__class__.__name__.find('BatchNorm') != -1:
        m.train(True)

def fix_bn(m):
    if m.__class__.__name__.find('BatchNorm') != -1:
        m.train(False)

# trainer tools
def merge_params(params, _params):
    return {**_params, **params}

def set_weight_file(model):
    model.weights_file = os.path.join('tmp', 'weights%d.pth' % (os.getpid()))
    check_dir('tmp')

def print_batch(e, epochs, b, batch_size, num_data, losses):
    ratio = 100.0 * ((b+1) * batch_size) / num_data
    print(chr(27) + "[2K", end='')
    print('\repochs #%d / %d | %d / %d (%6.2f %%) ' % (e + 1, epochs, (b+1) * batch_size, num_data, ratio), end='')
    print_loss(losses)
    sys.stdout.flush()

def time_format(t):
    m, s = divmod(t, 60)
    return '%d sec' % s if m==0 else '%d min %d sec' % (m, s)

def print_epoch(e, epochs, losses, epoch_time):
    print(chr(27) + "[2K", end='')
    print('\repochs #%d / %d ' % (e + 1, epochs), end='')
    print_loss(losses)
    print('\n - %s\n' % time_format(epoch_time))

def print_loss(losses):
    for r in losses:
        print('| %s = %.4f ' % (r, losses[r]), end='')

def detach_losses(losses):
    for loss in losses:
        losses[loss] = losses[loss].detach().item()
    return losses

def item_losses(losses):
    for loss in losses:
        losses[loss] = losses[loss].item()
    return losses

def merge_losses(losses_list):
    losses = losses_list[0]
    for loss in losses_list[0]:
        for losses_ in losses_list[1:]:
            losses[loss] += losses_[loss]
        losses[loss] /= len(losses_list)
    return losses

def add_losses(losses, new_losses):
    if losses:
        for loss in losses:
            losses[loss] += new_losses[loss]
        return losses
    else:
        return new_losses

def div_losses(losses, num):
    for loss in losses:
        losses[loss] = losses[loss]/float(num)
    return losses

def unpack_batch(batch_data, batch_i=[0,1]):
    return tuple_ind(batch_data, batch_i)

def get_sub_batch_data(batch_data, max_sub_size):
    batch_sizes = [data.shape[0] for data in batch_data]
    sub_num = max([batch_size/max_sub_size for batch_size in batch_sizes])
    sub_sizes = [int(batch_size/sub_num) for batch_size in batch_sizes]
    for b in range(int(np.ceil(sub_num))):
        yield tuple(data[b*sub_size:(b+1)*sub_size] for sub_size,data in zip(sub_sizes,batch_data))

def get_predictions(model, test_tensors, extra_inputs=[], max_sub_size=200):
    if type(test_tensors) not in [list, tuple]:
        test_tensors = [test_tensors]
    with torch.no_grad():
        predictions = []
        for i in np.arange(0, test_tensors[0].shape[0], max_sub_size):
            predictions.append(model(*[test_tensor[i:i+max_sub_size] for test_tensor in test_tensors], *extra_inputs))
        return torch.vstack(predictions)

def save_args(outM, args, postfix=''):
    save_json(outM.get_filename('args%s'%postfix, 'json'), args)

def get_font(fontsize=15):
    return {'weight' : 'normal', 'size': fontsize}

def curve_plot(outM, curve_name, history, reporters=None, ylim=None, fontsize=15, color='base'):
    colors = colors_names[color]
    iters = range(len(history['loss'])) 
    if reporters is None:
        reporters = list(history.keys())
    save_history = {}
    # loss
    plt.figure()
    for r in range(len(reporters)):
        if reporters[r] in history:
            np.nan_to_num(history[reporters[r]])
            plt.plot(iters, history[reporters[r]], colors[r%len(colors)], label=reporters[r])
    plt.grid(color='k', linestyle='-', linewidth=0.5)
    plt.xlabel('epochs',get_font(fontsize))
    plt.ylabel('epochs-loss',get_font(fontsize))
    plt.legend(loc="upper right",prop=get_font(fontsize))
    if not ylim:
        tmp = history['loss']
        tmp = np.nan_to_num(tmp, nan=0, posinf=0, neginf=0)
        y_max = np.max(tmp)
        for r in reporters:
            if r in history:
                y_max = np.max((y_max, np.max(np.nan_to_num(history[r], nan=0, posinf=0, neginf=0))))
                save_history[r] = history[r]
        plt.ylim((0, y_max))
    else:
        plt.ylim(ylim[0],ylim[1])
    plt.tick_params(labelsize=fontsize)
    plt.tight_layout()
    plt.savefig(outM.get_filename(curve_name, 'png'))
    with open(outM.get_filename(curve_name, 'json'),'w') as f:
        f.write(tojson(save_history))

def save_evaluate(output=None, name=None, head=None, varList=None, filename=None):
    # save evaluate result
    if filename is None:
        filename = os.path.join(check_dir(os.path.join('output', output)), name+'_evaluate.csv')
    if not(os.path.exists(filename)):
        eval_file=open(filename, 'w')
        eval_file.write(head)
    else:
        eval_file=open(filename, 'a')
    varList = list(map(lambda x: round(x,3) if isinstance(x, float) else x , varList))
    txt=','.join(list(map(lambda x:str(x),varList)))
    eval_file.writelines(txt+'\n')
    eval_file.close()
    print(head+txt)

def save_t_SNE(filename, Xs, labels, n_components=2, fontsize=15, color='base', legend=True, markersize=12):
    if color in colors_names:
        colors = colors_names[color]
    else:
        cmap=plt.get_cmap(color)
        gradients = np.linspace(0, 1, len(Xs))
        colors = [cmap(gradient) for gradient in gradients]
    nums = np.array([x.shape[0] for x in Xs]) if len(Xs[0].shape)>1 else np.array([1 for x in Xs])
    if type(Xs[0]) is not np.ndarray:
        Xs = tuple(x.cpu().numpy() for x in Xs)
    X  = np.vstack(Xs)
    X_embedded = TSNE(n_components=n_components).fit_transform(X)
    Xs_embedded = np.split(X_embedded, nums.cumsum()[:-1])
    plt.figure()
    for i,xs_embedded in zip(range(len(Xs)), Xs_embedded):
        # plt.plot(xs_embedded[:,0], xs_embedded[:,1], colors[i], label=labels[i])
        plt.scatter(xs_embedded[:,0], xs_embedded[:,1], c=colors[i%len(colors)], marker='.', label=labels[i], s=markersize)
    plt.tick_params(labelsize=fontsize)
    plt.tight_layout()
    plt.xlabel('z1',get_font(fontsize))
    plt.ylabel('z2',get_font(fontsize))
    if legend:
        plt.legend(loc="upper right",prop=get_font(fontsize))
    plt.savefig(filename)
    