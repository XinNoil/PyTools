import os, sys, time, torch, argparse, math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mtools import tojson, save_json, load_json,check_dir, colors_names, str2bool
from sklearn.manifold import TSNE
import matplotlib.animation as animation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_device(d):
    print('>> device: %s\n' % str(device))
    return torch.device("cuda:%d"%d if torch.cuda.is_available() else "cpu")

def tensor(x):
    return torch.tensor(x, dtype=torch.float, device=device)

def stack_mean(x):
    return torch.mean(torch.stack(x), dim=0)
    
gen_path = os.environ['DEEPPRINT_GEN_PATH']

def get_filename(args, name, file_extension='csv', file_type='output', by_exp_no=True):
    postfix = '%s_%s' % (name, args.exp_no) if by_exp_no else name
    if file_type == 'output':
        return os.path.join(get_out_dir(args), '%s_%s.%s' % (args.model, postfix, file_extension))
    elif file_type == 'generate':
        return os.path.join(get_gen_dir(args), '%s_%s.%s' % (args.data_name, postfix, file_extension))

def get_out_dir(args):
    out_dir = '%s_%s' % (args.data_name, args.data_ver)
    if hasattr(args, 'feature_mode'):
        out_dir = '%s_%s' % (out_dir, args.feature_mode)
    if hasattr(args, 'sub_output'):
        if args.sub_output is not None:
            out_dir = '%s_%s' % (out_dir, args.sub_output)
    return check_dir(os.path.join('output', args.output, out_dir))

def get_gen_dir(args):
    return check_dir(os.path.join(gen_path, args.data_ver, args.output))

def set_weight_file(model):
    weights_no=np.random.randint(0,10000)
    model.weights_file = os.path.join('tmp', 'weights%d.pth' % (weights_no))
    check_dir('tmp')

def get_exp_no(args, e):
    if hasattr(args, 'data_postfix'):
        if args.data_postfix is not None:
            return args.data_postfix + 'e' + str(e)
    return 'e' + str(e)

def get_start_exp_no(exp_no):
    return 0 if exp_no == '' else int(exp_no)

def get_model_params(model_params, default_model_params):
    for param in default_model_params:
        if param not in model_params:
            model_params[param] = default_model_params[param]
    return model_params

def get_param(dict_, param):
    return dict_[param].copy() if isinstance(dict_[param], list) else dict_[param]

def merge_params(params, _params):
    for param in _params:
        if param not in params:
            params[param] = _params[param]
    return params

# torch models tools
def get_layers(input_dim, layer_units, Linear = torch.nn.Linear):
    layers = torch.nn.ModuleList()
    layers.append(Linear(input_dim, layer_units[0]))
    for l in range(len(layer_units)-1):
        layers.append(Linear(layer_units[l], layer_units[l+1]))
    return layers

def initialize_model(model):
    for m in model.modules():
        if issubclass(type(m), torch.nn.Linear):
            torch.nn.init.uniform_(m.weight, -0.05, 0.05)
        # elif issubclass(type(m), (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        #     torch.nn.init.xavier_uniform(m.weight)

def reset_parameters(model):
    for m in model.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

def spectral_norm(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
        return torch.nn.utils.spectral_norm(m)
    else:
        return m

def get_parameters(model, names):
    return [{'params':model.__dict__[name].parameters()} for name in names]

def get_model_parameters(models):
    return [{'params':model.parameters()} for model in models]

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

def n2t(num, tensortype=torch.FloatTensor, gpu=True):
    return tensortype(num).to(device) if gpu else tensortype(num)

def t2n(tensor):
    return tensor.detach().cpu().numpy()

def print_batch(e, epochs, b, batch_size, num_data, losses):
    ratio = 100.0 * ((b+1) * batch_size) / num_data
    print(chr(27) + "[2K", end='')
    print('\repochs #%d / %d | %d / %d (%6.2f %%) ' % (e + 1, epochs, (b+1) * batch_size, num_data, ratio), end='')
    print_loss(losses)
    sys.stdout.flush()

def print_epoch(e, epochs, losses, epoch_time):
    print(chr(27) + "[2K", end='')
    print('\repochs #%d / %d ' % (e + 1, epochs), end='')
    print_loss(losses)
    print('\n - %s\n' % time_format(epoch_time))

def print_loss(losses):
    for r in losses:
        print('| %s = %.4f ' % (r, losses[r].item()), end='')

def save_model(model):
    save_json(get_filename(model.args, 'args', 'json'), model.args)
    torch.save(model.state_dict(), get_filename(model.args, 'model_%s' % model.name, 'pth'))
    if hasattr(model, 'model_params'):
        save_json(get_filename(model.args, 'model_%s' % model.name, 'json', by_exp_no=False), model.model_params)

def load_model(model, args=None):
    if not args:
        args = model.args
    # if os.path.exists(get_filename(args, 'model_%s' % model.name, 'json', by_exp_no=False)):
    #     if hasattr(model, 'set_model_params'):
    #         model.set_model_params(load_json(get_filename(args, 'model_%s' % model.name, 'json', by_exp_no=False)))
    #     else:
    #         model.set_model_params(load_json(get_filename(args, 'model_%s' % model.name, 'json', by_exp_no=False)))
    model.to(device)
    model.load_state_dict(torch.load(get_filename(args, 'model_%s' % model.name, 'pth')))

def time_format(t):
    m, s = divmod(t, 60)
    return '%d sec' % s if m==0 else '%d min %d sec' % (m, s)

def get_font(fontsize=15):
    return {'weight' : 'normal', 'size': fontsize}

def save_evaluate(output, name, head, varList):
    # save evaluate result
    filename=os.path.join(check_dir(os.path.join('output', output)), name+'_evaluate.csv')
    if not(os.path.exists(filename)):
        eval_file=open(filename, 'w')
        eval_file.write(head)
    else:
        eval_file=open(filename, 'a')
    varList = list(map(lambda x: round(x,3) if isinstance(x, float) else x , varList))
    txt=','.join(list(map(lambda x:str(x),varList)))
    eval_file.writelines(txt+'\n')
    eval_file.close()
    print(head)
    print(txt)

def save_t_SNE(args, Xs, labels, name='t-SNE', n_components=2, fontsize=15, color='base', legend=True, markersize=12, filename=None):
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
        plt.scatter(xs_embedded[:,0], xs_embedded[:,1], c=colors[i], marker='.', label=labels[i], s=markersize)
    plt.tick_params(labelsize=fontsize)
    plt.tight_layout()
    plt.xlabel('z1',get_font(fontsize))
    plt.ylabel('z2',get_font(fontsize))
    if legend:
        plt.legend(loc="upper right",prop=get_font(fontsize))
    if filename:
        plt.savefig(filename)
    else:
        plt.savefig(get_filename(args, name, 'png'))

def save_t_SNE_ani(args, Xs, labels, name='t-SNE', n_components=2, fontsize=15, color='base', legend=True):
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
    fig = plt.figure()
    plt.tick_params(labelsize=fontsize)
    plt.tight_layout()
    plt.xlabel('z1',get_font(fontsize))
    plt.ylabel('z2',get_font(fontsize))
    ims = []
    for i,xs_embedded in zip(range(len(Xs)), Xs_embedded):
        im = plt.scatter(xs_embedded[:,0], xs_embedded[:,1], c=colors[i], marker='.', label=labels[i]).findobj()
        ims.append(im)
    plt.savefig(get_filename(args, name, 'png'))
    ani = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=1000)
    ani.save(get_filename(args, name, 'gif'),writer='pillow')

def curve_plot(history, args, curve_name='curve', reporters=None, ylim=None, fontsize=15, color='base'):
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
        np.nan_to_num(tmp)
        y_max = np.max(tmp)
        for r in reporters:
            if r in history:
                y_max = np.max((y_max, np.max(history[r])))
                save_history[r] = history[r]
        plt.ylim((0, y_max))
    else:
        plt.ylim(ylim[0],ylim[1])
    plt.tick_params(labelsize=fontsize)
    plt.tight_layout()
    plt.savefig(get_filename(args, curve_name, 'png'))
    with open(get_filename(args, curve_name, 'json'),'w') as f:
        f.write(tojson(save_history))

def copy_params(s, t, inds=None, indt=None, reverse=False):
    with torch.no_grad():
        if not reverse:
            is_input = True
            for ls,lt in zip(s.modules(), t.modules()):
                if issubclass(type(ls), torch.nn.Linear):
                    if is_input and (inds is not None):
                        for si,ti in zip(inds, indt):
                            lt.weight[:,ti].copy_(ls.weight[:,si])
                        # lt.weight[:,indt].copy_(ls.weight[:,inds])
                        lt.bias.copy_(ls.bias)
                        is_input = False
                    else:
                        lt.weight.copy_(ls.weight)
                        lt.bias.copy_(ls.bias)
        else:
            for i,ls,lt in zip(range(len(list(s.modules()))), s.modules(), t.modules()):
                if issubclass(type(ls), torch.nn.Linear):
                    if i==(len(list(s.modules()))-1) and (inds is not None):
                        for si,ti in zip(inds, indt):
                            lt.weight[ti,:].copy_(ls.weight[si,:])
                        lt.bias[indt].copy_(ls.bias[inds])
                    else:
                        lt.weight.copy_(ls.weight)
                        lt.bias.copy_(ls.bias)

def get_sub_batch_data(batch_data, max_sub_size):
    batch_sizes = [data.shape[0] for data in batch_data]
    sub_num = max([batch_size/max_sub_size for batch_size in batch_sizes])
    sub_sizes = [int(batch_size/sub_num) for batch_size in batch_sizes]
    for b in range(int(np.ceil(sub_num))):
        yield tuple(data[b*sub_size:(b+1)*sub_size] for sub_size,data in zip(sub_sizes,batch_data))

def merge_losses(losses_list):
    losses = losses_list[0]
    for loss in losses_list[0]:
        for losses_ in losses_list[1:]:
            losses[loss] += losses_[loss]
        losses[loss] /= len(losses_list)
    return losses