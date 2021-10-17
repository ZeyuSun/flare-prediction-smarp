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

    # Conv3d weight has shape (out, in, D, H, W) = (64, 1, 1, 11, 11)
    filters = weight.detach().cpu().numpy()
    vmin = vmin or filters.min()
    vmax = vmax or filters.max()

    if filters.shape[1] != 1:
        # Only visualize in-channels = 1
        filters = filters[:, :1]

    if filters.shape[2] == 1: # Time = 1
        n = int(np.ceil(np.sqrt(len(filters))))
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
    else: # Time > 1
        N = filters.shape[0]
        T = filters.shape[2]
        widths = [3] * T + [0.4]
        heights = [3] * N
        fig = plt.figure(figsize=(T/2, N/3))
        gs = mpl.gridspec.GridSpec(ncols=len(widths), nrows=len(heights), figure=fig,
                                   width_ratios=widths, height_ratios=heights)
        gs.update(wspace=0.05, hspace=0.05)
        for i in range(N):
            for j in range(T):
                ax = fig.add_subplot(gs[i, j])
                ax.imshow(filters[i, 0, j, :, :],
                          cmap='gray', vmin=vmin, vmax=vmax)
                ax.axis('off')
        ax = fig.add_subplot(gs[:,-1])
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cb = mpl.colorbar.ColorbarBase(ax, cmap=plt.get_cmap('gray'), norm=norm)
        gs.tight_layout(fig)
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
    - Return bin_total
    - Add argument prob_pred_value
    - Return exactly n_bins, with potentially np.nan

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

    bin_true = np.bincount(binids, weights=y_true, minlength=n_bins)
    # np.array([0, 1]) / 0 = array([nan, inf])
    prob_true = np.array([m / n if n != 0 else np.nan
                          for m, n in zip(bin_true, bin_total)])

    if prob_pred_value == 'center':
        half_bin = 0.5 / n_bins
        prob_pred = np.arange(half_bin, 1, half_bin*2)
    elif prob_pred_value == 'mean':
        raise ValueError('Nobody use this...')
        bin_sums = np.bincount(binids, weights=y_prob, minlength=n_bins)
        prob_pred = np.array([m / n if n != 0 else np.nan
                              for m, n in zip(bin_sums, bin_total)])
    else:
        raise ValueError('prob_pred_value should be either "center" or "mean".')

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


def check_and_convert(y_true, y_prob):
    import torch
    # Convert type to numpy array or list
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    elif hasattr(y_true, 'values'): # pd.Series, pd.DataFrame, ...
        y_true = y_true.values

    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy()
    elif hasattr(y_prob, 'values'):
        y_prob = y_prob.values

    # Nasty
    #if isinstance(y_true[0], (int, np.int64, float, bool, np.bool_)):
    #    y_true = [y_true]
    #if isinstance(y_prob[0], (int, float, bool, np.bool_)):
    #    y_prob = [y_prob]
    try:
        y_true[0][0]
    except IndexError:
        y_true = [y_true]
    try:
        y_prob[0][0]
    except IndexError:
        y_prob = [y_prob]

    # Broadcast
    if len(y_true) == 1 and len(y_prob) > 1:
        y_true = y_true * len(y_prob)
    if len(y_true) > 1 and len(y_prob) == 1:
        y_prob = y_prob * len(y_true)
    assert len(y_true) == len(y_prob)
    assert all([len(a) == len(b) for a, b in zip(y_true, y_prob)])

    return y_true, y_prob


def draw_reliability_plot(
        y_true,
        y_prob,
        n_bins=10,
        offset=0,
        fig_ax_ax2=None,
        marker='o',
        color='C0',
        name=None,
    ):
    """
    """
    y_true, y_prob = check_and_convert(y_true, y_prob)

    draw_error = len(y_true) > 1

    # Compute reliability diagram
    clim = np.mean([np.mean(y_t) for y_t in y_true])
    prob_true = [None] * len(y_true)
    for i, (y_t, y_p) in enumerate(zip(y_true, y_prob)):
        prob_true[i], prob_pred, bin_total = calibration_curve(
            y_t, y_p, n_bins=n_bins)
    prob_true_mean = np.nanmean(prob_true, axis=0)
    if draw_error:
        prob_true_std = np.nanstd(prob_true, axis=0)
    else:
        # Uncertainty of prob with a uniform prior (Wheatland 2004)
        # Since len(y_true) is 1, bin_total is the hist of all data
        prob_true_std = np.sqrt(prob_true_mean * (1-prob_true_mean)/(bin_total + 3))

    if fig_ax_ax2 is not None:
        fig, ax, ax2 = fig_ax_ax2
    else:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax2 = ax.twinx()
    ax.plot([0,1], [0,1], ':', color='C7')
    ax.plot([0,1], [clim, clim], ':', color='C7')
    ax.plot([0,1], [clim/2, (1+clim)/2], ':', color='C7')

    # Reliability diagram
    line, = ax.plot(
       prob_pred,
       prob_true_mean,
    ) # connect the errorbars
    line, _, _ = ax.errorbar(
        prob_pred + offset,
        prob_true_mean,
        yerr=prob_true_std,
        marker=marker,
        linestyle='',
        color=line.get_color(),
        label=name,
    )
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xticks(np.linspace(0,1,11), minor=True)
    ax.set_yticks(np.linspace(0,1,11), minor=True)
    #ax.set_aspect('equal')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Observed Frequency')

    # Histogram
    ax2.set_xlim([0,1])
    #ax2.set_ylim(0, None) # Passing top=None but it is still [0, 1]
    #ax2.set_yscale('log')
    #ax2.set_aspect('auto')  # equal aspect ratio won't work for twinx.
    # "aspect" in matplotlib always refers to the data, not the axes box.
    ax2.plot(prob_pred + offset, bin_total, '_', mew=3, ms=12) # ax2 has its own color cycle
    # plt.rcParams['lines.markersize'], plt.rcParams['lines.markeredgewidth'] = (6, 1)
    ax2.set_ylabel('Number of samples')

    plt.tight_layout() # the figsize (canvas size) remains square, and the axis gets squeezed and become thinner because of the two ylabels
    squarify(fig)
    return fig


def draw_roc(
        y_true,
        y_prob,
        fpr_grid=None,
        fig_ax=None,
        name=None,
    ):
    """
    TODO: y_true could have different elements in each row.
    """
    from sklearn.metrics import roc_curve, auc

    y_true, y_prob = check_and_convert(y_true, y_prob)

    draw_error = len(y_true) > 1

    # Calculate ROC and AUC
    fpr_grid = fpr_grid or np.linspace(0, 1, 501)
    tpr_grid = [None] * len(y_true)
    aucs = [None] * len(y_true)
    for i, (y_t, y_p) in enumerate(zip(y_true, y_prob)):
        fpr, tpr, thresholds = roc_curve(y_t, y_p)
        tpr_grid[i] = np.interp(fpr_grid, fpr, tpr)
        aucs[i] = auc(fpr, tpr)
    tpr_mean = np.mean(tpr_grid, axis=0)
    tpr_std = np.std(tpr_grid, axis=0)
    auc_mean = auc(fpr_grid, tpr_mean)
    auc_std = np.std(aucs)

    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(figsize=(5, 4.5))

    # No-skill line
    ax.plot([0, 1], [0, 1], ls=':', color='C7')

    # ROC
    if draw_error:
        label = r'AUC = %0.3f $\pm$ %0.3f' % (auc_mean, auc_std)
    else:
        label = r'AUC = %0.3f' % auc_mean
    if name is not None:
        label = name + ' (' + label + ')'
    line, = ax.plot(
        fpr_grid,
        tpr_mean,
        label=label,
        #color= automatic
    )

    # Error (optional)
    if draw_error:
        ax.fill_between(
            fpr_grid,
            np.maximum(tpr_mean - tpr_std, 0),
            np.minimum(tpr_mean + tpr_std, 1),
            color=line.get_color(),
            alpha=0.2,
        )

    # Layout
    ax.set(xlim=[0, 1],
           ylim=[0, 1],
           xlabel='FAR', # 'False Positive Rate'
           ylabel='POD') # 'True Positive Rate'
    ax.set_xticks(np.linspace(0,1,11), minor=True)
    ax.set_yticks(np.linspace(0,1,11), minor=True)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')

    plt.tight_layout()  # avoid xlabels being cut off, or use bbox_inches='tight' in savefig

    # Ideally, return (fig, ax)
    # for backward compatibility, we do not return ax.
    # use fig.axes[-1] to get the last axes
    return fig


def draw_tpr_fpr(y_true, y_prob):
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    thresholds[0] = thresholds[0] - 1 + 0.001
    fig, ax = plt.subplots(figsize=(3.75,3))
    ax.plot(thresholds, tpr, label='POD')
    ax.plot(thresholds, fpr, label='FAR')
    ax.set(xlabel='Threshold',
           ylabel='Metric')
    ax.set_xticks(np.linspace(0,1,11), minor=True)
    ax.legend()
    ax.grid()
    plt.tight_layout()
    return fig


def draw_ssp(
        y_true,
        y_prob,
        thresholds=None,
        scores=None,
        fig_ax=None,
        name=None,
    ):
    scores = scores or ['tss', 'hss']
    from sklearn.metrics import roc_curve

    y_true, y_prob = check_and_convert(y_true, y_prob)

    draw_error = len(y_true) > 1

    tss = [None] * len(y_true)
    hss = [None] * len(y_true)
    thresholds = thresholds or np.linspace(0, 1, 501)
    for i, (y_t, y_p) in enumerate(zip(y_true, y_prob)):
        _fpr, _tpr, _thresh = roc_curve(y_t, y_p)
        _fpr, _tpr, _thresh = _fpr[1:][::-1], _tpr[1:][::-1], _thresh[1:][::-1]
        # remove added point. revert _thresh so that it is increasing
        fpr = np.interp(thresholds, _thresh, _fpr)
        tpr = np.interp(thresholds, _thresh, _tpr)

        tss[i] = tpr - fpr

        P = y_t.sum() #np.sum(y_true) don't use np.sum. Won't take sum() received an invalid combination of arguments - got (out=NoneType, axis=NoneType, ), but expected one of:
        N = len(y_t) - P
        FP, TP = N * fpr, P * tpr
        TN, FN = N - FP, P - TP
        hss[i] = 2 * (TP * TN - FN * FP) / (P * (FN+TN) + (TP+FP) * N)
    tss_mean = np.mean(tss, axis=0)
    tss_std = np.std(tss, axis=0)
    hss_mean = np.mean(hss, axis=0)
    hss_std = np.std(hss, axis=0)

    
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    suffix = '' if name is None else f' ({name})'
    if name is None:
        tss_label = 'TSS'
        hss_label = 'HSS'
    else:
        tss_label = f'{name} (TSS)'
        hss_label = f'{name} (HSS)'
    if 'tss' in scores:
        line_tss, = ax.plot(thresholds, tss_mean, label=tss_label)
    if 'hss' in scores:
        line_hss, = ax.plot(thresholds, hss_mean, label=hss_label)
    if draw_error:
        if 'tss' in scores:
            ax.fill_between(
                thresholds,
                np.maximum(tss_mean - tss_std, 0),
                np.minimum(tss_mean + tss_std, 1),
                color=line_tss.get_color(),
                alpha=0.2,
            )
        if 'hss' in scores:
            ax.fill_between(
                thresholds,
                np.maximum(hss_mean - hss_std, 0),
                np.minimum(hss_mean + hss_std, 1),
                color=line_hss.get_color(),
                alpha=0.2,
            )
    ax.legend(loc='lower center')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    #ax.set_ylim(-0.05, 1.05)
    #ax.set_xlim(-0.05, 1.05)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal') # make sure the plane looks like ROC
    plt.tight_layout()
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
