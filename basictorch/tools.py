import os, sys, time, torch, argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mtools import tojson, save_json, load_json,check_dir

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('>> device: %s\n' % str(device))
gen_path = os.environ['DEEPPRINT_GEN_PATH']

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def get_filename(args, name, file_extension='csv', file_type='output', by_exp_no=True):
    postfix = '%s_%s' % (name, args.exp_no) if by_exp_no else name
    if file_type == 'output':
        return os.path.join(get_out_dir(args), '%s_%s.%s' % (args.model, postfix, file_extension))
    elif file_type == 'generate':
        return os.path.join(get_gen_dir(args), '%s_%s.%s' % (args.data_name, postfix, file_extension))

def get_out_dir(args): 
    return check_dir(os.path.join('output', args.output, '%s_%s_%s' % (args.data_name, args.data_ver, args.feature_mode) if hasattr(args, 'feature_mode') else '%s_%s' % (args.data_name, args.data_ver)))

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

def get_layers(input_dim, layer_units, Linear = torch.nn.Linear):
    layers = torch.nn.ModuleList()
    layers.append(Linear(input_dim, layer_units[0]))
    for l in range(len(layer_units)-1):
        layers.append(Linear(layer_units[l], layer_units[l+1]))
    return layers

def initialize_model(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.uniform_(m.weight, -0.05, 0.05)

def n2t(num):
    return torch.FloatTensor(num).to(device)

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
    if os.path.exists(get_filename(args, 'model_%s' % model.name, 'json', by_exp_no=False)):
        model.set_model_params(load_json(get_filename(args, 'model_%s' % model.name, 'json', by_exp_no=False)))
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

def curve_plot(history, args, curve_name='curve', reporters=None, ylim=None, fontsize=15):
    iters = range(len(history['loss'])) 
    if reporters is None:
        reporters = list(history.keys())
    colors = ['k','g','r','b','c','y','m','grey','brown','orange','olive','purple','pink']
    save_history = {}
    # loss
    plt.figure()
    for r in range(len(reporters)):
        if reporters[r] in history:
            np.nan_to_num(history[reporters[r]])
            plt.plot(iters, history[reporters[r]], colors[r], label=reporters[r])
    plt.grid(color='k', linestyle='-', linewidth=0.5)
    plt.xlabel('epochs',get_font(fontsize))
    plt.ylabel('epochs-loss',get_font(fontsize))
    plt.legend(loc="upper right",prop=get_font(fontsize))
    if not ylim:
        y_max = np.max(history['loss'])
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

def copy_params(s, t, inds=None, indt=None):
    for l1,l2 in zip(s.modules(), t.modules()):
        if issubclass(type(l1), torch.nn.Linear):
            if l1.weight.shape == l2.weight.shape:
                t.weight = s.weight
                t.bias = s.bias
            else:
                for si,ti in zip(inds, indt):
                    t.weight[ti] = s.weight[si]
                t.bias = s.bias