import os
import numpy as np
import matplotlib.pyplot as plt

def plot_test_errs(data_path, date, i, cdns, test_errs, dlim = [0, 5]):
    if not os.path.exists(os.path.join(data_path, 'test_errs', 'Date',   date[:-2])):
        os.mkdir(os.path.join(data_path, 'test_errs', 'Date',   date[:-2]))
    if not os.path.exists(os.path.join(data_path, 'test_errs', 'Device', date[:-2])):
        os.mkdir(os.path.join(data_path, 'test_errs', 'Device', date[:-2]))
    data_path1 = os.path.join(data_path, 'test_errs', 'Date',   date[:-2], '%s-D%d'%(date, i+1))
    data_path2 = os.path.join(data_path, 'test_errs', 'Device', date[:-2], 'D%d-%s'%(i+1, date))
    
    # csvwrite('%s_err.csv'%data_path1, np.hstack((cdns, test_errs[:, np.newaxis])))
    # csvwrite('%s_err.csv'%data_path2, np.hstack((cdns, test_errs[:, np.newaxis])))
    xlim = (np.min(cdns[:,0])-1, np.max(cdns[:,0])+1)
    ylim = (np.min(cdns[:,1])-1, np.max(cdns[:,1])+1)
    plt.figure(figsize=(16, 6))
    fig,ax = plt.subplots(2, 1)

    ax0 = ax[0].scatter(cdns[:, 0], cdns[:, 1], marker='o', c=test_errs, cmap='jet')
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    fig.colorbar(ax0, ax=ax[0], label='Error distance (m)')

    cdns = np.vstack((cdns, np.array([[-100, -100],[-100, -100]])))
    test_errs = np.hstack((test_errs, np.array(dlim)))
    test_errs[test_errs<dlim[0]] = dlim[0]
    test_errs[test_errs>dlim[1]] = dlim[1] 
    ax1 = ax[1].scatter(cdns[:, 0], cdns[:, 1], marker='o', c=test_errs, cmap='jet')
    ax[1].set_xlim(xlim)
    ax[1].set_ylim(ylim)
    fig.colorbar(ax1, ax=ax[1], label='Error distance (m)')
    plt.savefig('%s_err.png'%data_path1)
    plt.savefig('%s_err.png'%data_path2)