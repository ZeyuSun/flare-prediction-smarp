import numpy as np
import matplotlib.pyplot as plt


def get_log_intensity(class_str):
    if class_str == '' or class_str == 'Q':
        return -9. # torch.floor not implemented for Long

    class_map = {
        'A': -8.,
        'B': -7.,
        'C': -6.,
        'M': -5.,
        'X': -4.,
    }
    a = class_map[class_str[0]]
    b = np.log10(float(class_str[1:]))
    return a + b

def get_flare_class(log_intensity, class_only=False):
    class_map = {
        -8: 'A',
        -7: 'B',
        -6: 'C',
        -5: 'M',
        -4: 'X',
    }

    floor = np.floor(log_intensity)
    floor = int(min(floor, -4))
    if floor < -8:
        return 'Q'

    letter = class_map[floor]
    if class_only:
        return letter

    level = 10 ** (log_intensity - floor)
    level = f'{level:.1f}'
    return letter + level

def plot_flare_history(flares, indices, session_time, harp_start):
    from datetime import timedelta

    palette = {'A': 'black', 'B': 'blue', 'C': 'green', 'M': 'red', 'X': 'red'}

    print(flares)
    print(harp_start)

    noaa = flares['noaa_ar_num'].iloc[0]
    t = flares['start_time']
    x = flares['class'].apply(get_log_intensity)
    c = flares['class'].apply(lambda s: palette[s[0]])

    fig = plt.figure()
    plt.scatter(t, x, c=c)
    plt.title('AR {:d}'.format(int(noaa)))
    fig.autofmt_xdate()
    xlim = [harp_start, t.iloc[-1]]
    dt = (xlim[1] - xlim[0]) * 0.1
    xlim = [xlim[0], xlim[1] + dt]
    plt.xlim(xlim)
    for i in indices:
        plt.hlines(get_log_intensity(flares.loc[i, 'class']),
                   flares.loc[i, 'start_time'] - session_time,
                   flares.loc[i, 'start_time'])
    plt.show()

def fig2rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgb_string = fig.canvas.tostring_rgb()
    image = np.frombuffer(rgb_string, dtype=np.uint8).reshape(h,w,3).transpose(2,0,1)
    plt.close()  # close the window to save memory

    return image

def draw_conv2d_weight(weight, vmin=None, vmax=None):
    import matplotlib as mpl

    # Conv3d weight has shape (64, 1, 1, 11, 11)
    filters = weight.detach().cpu().numpy()
    n = int(np.ceil(np.sqrt(len(filters))))
    vmin = vmin or filters.min()
    vmax = vmax or filters.max()

    widths = [3] * n + [0.4]
    heights = [3] * n
    fig = plt.figure(figsize=(10,10))
    gs = mpl.gridspec.GridSpec(ncols=len(widths), nrows=len(heights), figure=fig,
                               width_ratios=widths, height_ratios=heights)
    gs.update(wspace=0.05, hspace=0.05)
    for i in range(n):
        for j in range(n):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(filters[i*n+j, 0, 0, :, :],
                      cmap='gray', vmin=vmin, vmax=vmax)
            ax.axis('off')
    ax = fig.add_subplot(gs[:,-1])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=plt.get_cmap('gray'), norm=norm)
    #gs.tight_layout(fig)
    plt.close()
    return fig

def draw_confusion_matrix(matrix, labels=None):
    from itertools import product

    d = 1.2 * matrix.shape[0] + 0.6  # 2->3, 6->7.8
    fig = plt.figure(figsize=(d,d))
    ax = fig.add_subplot(111)
    #ax.set_title('Confusion matrix')

    cm = np.asarray(matrix).astype(int)
    num_classes = cm.shape[0]
    ticks = np.arange(num_classes)
    ax.imshow(cm, cmap='Oranges')

    ax.set_xlabel('Predicted')
    ax.set_xticks(ticks)
    if labels is not None:
        ax.set_xticklabels(labels)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    ax.set_ylabel('Actual')
    ax.set_yticks(ticks)
    if labels is not None:
        ax.set_yticklabels(labels)
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in product(range(num_classes), range(num_classes)):
        text = '.' if cm[i,j] == 0 else '{}'.format(cm[i,j])
        ax.text(j, i, text, horizontalalignment="center",
                verticalalignment="center", color="C0", fontweight='bold', fontsize=16)

    plt.tight_layout()
    return fig

def _column_or_1d(y):
    y = np.asarray(y)
    shape = np.shape(y)
    if len(shape) == 1 or len(shape) == 2 and shape[1] == 1:
        return np.ravel(y)
    raise ValueError(
        'y should be 1d array, '
        'got an array of shape {} instead.'.format(shape))

def calibration_curve(y_true, y_prob, *, n_bins=5, prob_pred_value='center'):
    """A modified version of sklearn.calibration.calibration_curve

    Args:
        y_true: Array-like of shape (n_samples,)
            True class labels taking values 0 or 1.
        y_prob: Array-like of shape (n_samples,)
            Probabilities of positive class.
        n_bins: int, default=5
            Number of bins to discretize the [0,1] interval. Bins with no
            samples will not be returned, thus the returned arrays may have
            less than `n_bins` entries.
        prob_pred_value: str, {'center', 'mean'}, default='center'
            The value type returned by prob_pred. 'center' gives the middle
            point of each bin, while 'mean' gives the mean predicted
            probability in each bin.

    Returns:
        prob_true: ndarray of shape(n_bins,) or smaller
        prob_pred: ndarray of shape(n_bins,) or smaller
        bin_total: ndarray of shape(n_bins,) or smaller
    """
    y_true = _column_or_1d(y_true)
    y_prob = _column_or_1d(y_prob)
    if y_true.shape != y_prob.shape:
        raise ValueError('y_true and y_prob should be of same length.')

    if not all([y in {0,1} for y in y_true]):
        raise ValueError("y_true should contain value 0 or 1")

    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    bin_total = np.bincount(binids, minlength=n_bins)
    nonzero = bin_total != 0

    bin_true = np.bincount(binids, weights=y_true, minlength=n_bins)
    prob_true = bin_true[nonzero] / bin_total[nonzero]

    if prob_pred_value == 'center':
        half_bin = 0.5 / n_bins
        prob_pred = np.arange(half_bin, 1, half_bin*2)[nonzero]
    elif prob_pred_value == 'mean':
        bin_sums = np.bincount(binids, weights=y_prob, minlength=n_bins)
        prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    else:
        raise ValueError('prob_pred_value should be either "center" or "mean".')

    bin_total = bin_total[nonzero]

    return prob_true, prob_pred, bin_total

def squarify(fig):
    w, h = fig.get_size_inches()
    if w > h:
        t = fig.subplotpars.top
        b = fig.subplotpars.bottom
        axs = h*(t-b)
        l = (1.-axs/w)/2
        fig.subplots_adjust(left=l, right=1-l)
    else:
        t = fig.subplotpars.right
        b = fig.subplotpars.left
        axs = w*(t-b)
        l = (1.-axs/h)/2
        fig.subplots_adjust(bottom=l, top=1-l)

def draw_reliability_plot(y_true, y_prob, n_bins=5):
    clim = y_true.double().mean() # np.mean's argument can't be torch.Tensor
    prob_true, prob_pred, bin_total = calibration_curve(y_true, y_prob,
                                                        n_bins=n_bins)
    sigma = np.sqrt(prob_true * (1-prob_true)/(bin_total + 3))
    #sigma = np.sqrt(prob_true * ((1-prob_true)/bin_total))
    fig, ax1 = plt.subplots(figsize=(3.75,3))
    ax1.plot(prob_pred, prob_true)
    ax1.errorbar(prob_pred, prob_true, yerr=sigma,
                 marker='o', color='C0', linestyle='')
    ax1.plot([0,1], [0,1], ':', color='C7')
    ax1.plot([0,1], [clim, clim], ':', color='C7')
    ax1.plot([0,1], [clim/2, (1+clim)/2], ':', color='C7')
    ax1.set_xlim([0,1])
    ax1.set_ylim([0,1])
    ax1.set_xticks(np.linspace(0,1,11), minor=True)
    ax1.set_yticks(np.linspace(0,1,11), minor=True)
    #ax1.set_aspect('equal')
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Observed Frequency', color='C0')

    ax2 = ax1.twinx()
    ax2.set_xlim([0,1])
    #ax2.set_aspect('auto')  # equal aspect ratio won't work for twinx.
    # "aspect" in matplotlib always refers to the data, not the axes box.
    ax2.plot(prob_pred, bin_total, 's--', color='C1')  # red, square
    ax2.set_ylabel('Number of samples', color='C1')

    plt.tight_layout() # the figsize (canvas size) remains square, and the axis gets squeezed and become thinner because of the two ylabels
    squarify(fig)
    return fig

def draw_roc(y_true, y_prob):
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(3.75,3))
    ax.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc(fpr, tpr)))
    ax.plot([0,1], [0,1], ls=':')
    ax.set(xlim=[0,1], ylim=[0,1.03],
           xlabel='False Positive Rate',
           ylabel='True Positive Rate')
    ax.set_xticks(np.linspace(0,1,11), minor=True)
    ax.set_yticks(np.linspace(0,1,11), minor=True)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')

    plt.tight_layout()  # avoid xlabels being cut off, or use bbox_inches='tight' in savefig
    return fig

def draw_ssp(y_true, y_prob):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fpr, tpr, thresholds = fpr[1:], tpr[1:], thresholds[1:] # remove added point
    P = y_true.sum() #np.sum(y_true) don't use np.sum. Won't take sum() received an invalid combination of arguments - got (out=NoneType, axis=NoneType, ), but expected one of:
    N = len(y_true) - P
    FP, TP = N * fpr, P * tpr
    TN, FN = N - FP, P - TP
    TSS = tpr - fpr
    HSS2 = 2 * (TP * TN - FN * FP) / (P * (FN+TN) + (TP+FP) * N)

    fig, ax = plt.subplots(figsize=(3.75,3))
    ax.plot(thresholds, TSS, label='TSS')
    ax.plot(thresholds, HSS2, label='HSS2')
    ax.legend()
    return fig


def plot_prediction_curve(noaa_ar: int,
                          times: list,
                          y_prob: list,
                          y_true: list,
                          filename=None):
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt

    goes_df = pd.read_csv('/home/zeyusun/SOLSTICE/goes/GOES_HMI.csv')
    goes_df = goes_df[goes_df['noaa_active_region'] == noaa_ar]
    goes_df['peak_time'] = goes_df['peak_time'].apply(pd.to_datetime)
    goes_df = goes_df[(goes_df['peak_time'] >= times[0]) &
                      (goes_df['peak_time'] <= times[-1])]
    goes_df['intensity'] = goes_df['goes_class'].apply(get_log_intensity)

    fig, ax1 = plt.subplots(figsize=(8,3))
    color = goes_df['intensity'].apply(np.floor) #.astype(int) # int(-3.2)=-3 #round towards 0
    norm = matplotlib.colors.Normalize(vmin=-8, vmax=-4)
    ax1.scatter(goes_df['peak_time'], goes_df['intensity'], c=color,
                cmap='Reds', norm=norm)
    ax1.set_ylabel('Log intensity')
    ax1.set_ylim([-7,-3])
    ax1.set_yticks([-7, -6, -5, -4, -3])
    ax1.axhline(-5, linestyle=':')
    from matplotlib.ticker import MaxNLocator
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twinx()
    ax2.plot(times, y_prob, 'o-', label='prediction')
    ax2.plot(times, y_true, label='ground truth')
    ax2.legend()
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Probability')
    # xlim (to compare between different ARs, draw 120hr / 5days before the peak time)
    plt.xticks(rotation=20, ha='right') # ax.set_xticks needs a ticks argument

    plt.tight_layout() # ow rotated xticks will be outside the canvas
    filename = filename or 'temp.png'
    plt.savefig(filename)
    #plt.show()

if __name__ == '__main__':
    # # Test draw_reliability_plot
    # y_true = np.array([0, 0, 1, 0, 1])
    # y_prob = np.array([0., 0.1, 0.2, 0.8, 0.9])
    # prob_true, prob_pred, bin_total = calibration_curve(y_true, y_prob, n_bins=2)
    # draw_reliability_plot(y_true, y_prob, n_bins=2)
    # #plt.savefig('1.png')
    # plt.show()
    #
    # draw_roc(y_true, y_prob)
    # #plt.savefig('2.png')
    # plt.show()
    #
    # draw_ssp(y_true, y_prob)
    # plt.show()

    # Test get_flare_class
    for i in np.linspace(-10,-2,20):
        c = get_flare_class(i)
        ihat = get_log_intensity(c)
        print(i, c, ihat)
