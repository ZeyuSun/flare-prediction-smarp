import os
from glob import glob
from datetime import datetime, timedelta
from ipdb import set_trace as breakpoint
import numpy as np
import pandas as pd
import xarray as xr
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm.notebook import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
import sunpy
from captum.attr import IntegratedGradients, Saliency, DeepLift, GuidedBackprop, GuidedGradCam, Deconvolution, NoiseTunnel, LRP, LayerGradCam, LayerLRP

from arnet.utils import get_layer, get_log_intensity
from arnet.dataset import ActiveRegionDataModule, ActiveRegionDataset
from arnet.transforms import get_transform
from cotrain_helper import get_learner_by_query


harpnum_to_noaa = np.load('harpnum_to_noaa.npy', allow_pickle=True).item()
goes_df = pd.read_csv('/home/zeyusun/SOLSTICE/goes/GOES_HMI.csv', index_col=0)
goes_df['start_time'] = goes_df['start_time'].apply(pd.to_datetime)
goes_df['intensity'] = goes_df['goes_class'].apply(get_log_intensity)
goes_df['flare_class'] = goes_df['goes_class'].str[0]


def get_heatmap(algorithm, learner, input, target='negate', baselines='zero'):
    """
    TODO: heatmaps should only have 1 channels
    """
    model = learner.model
    model.eval()
    model.zero_grad()
    if target == 'negate': # negate should be taken care of inside function.
        logit = model(input)
        pred = logit[:, 1] > logit[:, 0]
        target = 1 - pred.to(int)
    else: # target is a tensor
        target = torch.tensor(target) # target may be an int, e.g., 1
        target = target.to(input.device)

    if baselines == 'zero':
        baselines = input * 0

    if algorithm == 'Original':
        heatmap = input
    elif algorithm == 'Saliency':  # (Simonyan 2013)
        saliency = Saliency(model)
        heatmap = saliency.attribute(
            input,
            target=target,
        )
    elif algorithm == 'Deconvolution':  # (Zeiler 2013)
        deconv = Deconvolution(model)
        heatmap = deconv.attribute(
            input,
            target=target,
        )
    elif algorithm == 'GuidedBackprop':  # (Springenberg 2014)
        guided_backprop = GuidedBackprop(model)
        heatmap = guided_backprop.attribute(
            input,
            target=target,
        )
    elif algorithm[:12] == 'LayerGradCam':  # (Selvaraju 2016)
        if algorithm == 'LayerGradCam':
            algorithm += '-5' # last layer is 5
        n_layer = int(algorithm.split('-')[1])
        layer = get_layer(model, f'convs.conv{n_layer}')
        layer_gradcam = LayerGradCam(model, layer)
        heatmap = layer_gradcam.attribute(
            input,
            target=target,
            relu_attributions=True,
        )
        # up to 5D input is supported: (B, C, [D], [H], W)
        # the `size` argument does not include B and C
        heatmap = torch.nn.functional.interpolate(
            heatmap,
            size=input[0, 0].shape,
            mode='trilinear',
            align_corners=False,
        )
    elif algorithm == 'ArnetGradCam':
        from arnet.utils import GradCAM
        target_layers = ['convs.conv5']
        gradcam = GradCAM(
            model,
            target_layers,
            data_mean=0,
            data_std=1,
        )
        overlapped, image, heatmap, preds = gradcam(
            input,
            labels=target,
        )
    elif algorithm == 'GuidedGradCam':  # (Selvaraju 2016)
        guided_gradcam = GuidedGradCam(model, model.convs.conv4) #conv5)
        heatmap = guided_gradcam.attribute(
            input,
            target=target,
        )
    elif algorithm == 'IntegratedGradients':  # (Sundararajan 2017)
        ig = IntegratedGradients(model)
        heatmap, delta = ig.attribute(
            input,
            target=target,
            baselines=baselines,
            return_convergence_delta=True
        )
        print('Approximation delta: ', abs(delta))
    elif algorithm == 'NoiseTunnel':
        ig = IntegratedGradients(model)
        nt = AttrAlgo(ig)
        heatmap = nt.attribute(
            input,
            target=target,
            baselines=baselines,
            nt_type='smoothgrad_sq',
            nt_samples=100,
            stdevs=0.2
        )
    elif algorithm == 'DeepLift':  # (Shrikumar 2017)
        deeplift = DeepLift(model)
        heatmap = deeplift.attribute(
            input,
            target=target,
            baselines=baselines,
        )
    elif algorithm == 'LRP':
        # Does not support Conv3d
        lrp = LRP(model)
        heatmap = lrp.attribute(
            input,
            target=target,
        )
    elif algorithm == 'LayerLRP':
        # Does not support Conv3d
        layer_lrp = LayerLRP(model, model.convs.conv5)
        heatmap = layer_lrp.attribute(
            input,
            target=target,
        )
        heatmap = torch.nn.functional.interpolate(
            heatmap,
            size=input[0, 0].shape,
            mode='trilinear',
            align_corners=False,
        )
    else:
        raise
    # heatmap.shape = (N, C=1, T=1, H, W)
    #heatmap = heatmap.cpu().detach().numpy() # should return the same dtype and shape as input
    return heatmap


def get_heatmaps_from_df(df, algorithms,
                         num_frames=1,
                         num_frames_after=0,
                         target_type='pos',
                         baseline_type='zero'):
    """
    Args:
        df: has columns 'model_query'. Index 0, 1, ...
        algorithms (List[str]): list of attribution algortihm names.
        target_type (str):
            'pos': positive class
            'negate': negate the prediction
            'label': same with the label
            other: pass directly
        baseline_type (str):
            'zero': (default) zero baseline.
            'first': first row with the same model_query. User has to make sure
                all rows in model_query subdf belong to the same active region.
            other: pass directly.

    Returns:
        heatmaps: list of attribution heatmaps in the order of df.
            Each heatmap is of shape (C, T, H, W) = (1, 1, 128, 128).

    Note:
    * This function involves no concept of animation. df in, heatmaps out.
    * I once implemented for each entry in df, expand 24 hr forward and backward
    in time. It relies on the ActiveRegionDataset's functionality to retrieve
    more data, as opposed to relying on df. Cons: unnecessarily convoluted (just
    use df with consecutive entries); prone to repetitive computation due to
    video overlapping for consecutive entries; baselines has more than 1 frame.
    """
    assert num_frames == 1 # See Note
    assert num_frames_after == 0 # See Note
    heatmaps = {a: [None] * len(df) for a in algorithms}
    df = df.reset_index(drop=True)
    for model_query, subdf in df.groupby('model_query'):
        # # hotfix begin
        # fields = model_query.split('/')
        # fields.insert(-1, '0')
        # model_query = '/'.join(fields)
        # # hotfix end

        learner = get_learner_by_query(model_query).to('cuda')
        cfg = learner.cfg
        transform = Compose([get_transform(name, cfg)
                             for name in cfg.DATA.TRANSFORMS] +
                            [lambda video: video.to('cuda')])
        dataset = ActiveRegionDataset(
            subdf,
            features=['MAGNETOGRAM'],
            num_frames=num_frames,
            num_frames_after=num_frames_after,
            transform=transform
        )
        dataloader = DataLoader(
            dataset,
            batch_size=16 if 'IntegratedGradients' not in algorithms else 2, # at most 4 if IG in algorithms
            #shuffle=False,
            #num_workers=8,
            #pin_memory=False,
            #drop_last=False,
        )

        for algorithm in tqdm(algorithms):
            hs = []
            for videos, sizes, labels, meta in dataloader:
                if target_type == 'pos':
                    target = torch.ones(labels.shape, dtype=int)
                elif target_type == 'neg':
                    target = torch.zeros(labels.shape, dtype=int)
                elif target_type == 'negate':
                    target = 'negate'
                elif target_type == 'label':
                    target = labels
                else: # option to pass directly
                    target = target_type
                if baseline_type == 'first':
                    first_video = dataset[0][0].unsqueeze(0) # first dimensions is to be broadcasted if batch_size > 1
                    baselines = first_video
                elif baseline_type == 'zero':
                    baselines = 'zero' # 0 input
                else:
                    baselines = baseline_type
                h = [get_heatmap(algorithm,
                                 learner,
                                 videos[:, :, [t]],
                                 target=target,
                                 baselines=baselines
                                ).cpu().detach().numpy()
                     for t in range(videos.shape[2])]
                h = np.concatenate(h, axis=2)
                # h.shape == videos.shape = [N, C=1, T, H, W]
                hs.append(h)
            hmaps = np.concatenate(hs) # along the batch dimension

            for i, m in zip(subdf.index, hmaps):
                heatmaps[algorithm][i] = m
    return heatmaps


def resize_heatmaps(heatmaps, df):
    for i in range(len(df)):
        prefix = df['prefix'].iloc[i]
        arpnum = df['arpnum'].iloc[i]
        t_end = pd.to_datetime(df['t_end'].iloc[i])
        data, header = load_data_and_header(prefix, arpnum, t_end)

        for a in heatmaps:
            assert heatmaps[a][i].shape[:2] == (1,1)
            if a == 'Original':
                heatmaps[a][i] = np.expand_dims(data, axis=(0,1))
            else:
                heatmap = resize(
                    heatmaps[a][i][0, 0],
                    data.shape,
                    order=2
                )
                heatmaps[a][i] = np.expand_dims(heatmap, axis=(0,1))
    return heatmaps


def get_heatmaps_synthetic(Z_list, algorithms, learner):
    dataset = [(torch.tensor(np.expand_dims(Z, axis=(0,1)))
                .to(learner.device)
                .to(torch.float32))
               for Z in Z_list]

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        #num_workers=0,
    )

    heatmaps = {}
    for algorithm in tqdm(algorithms):
        hs = []
        for videos in tqdm(dataloader):
            h = get_heatmap(algorithm, learner, videos).cpu().detach().numpy()
            hs.append(h)
        heatmaps[algorithm] = np.concatenate(hs)
    return heatmaps


def get_t_steps(t_now: str, num_frames=16, num_frames_after=15):
    dt = timedelta(minutes=96)
    t_now = datetime.strptime(t_now, '%Y-%m-%d %H:%M:%S')
    t_start = t_now - dt * (num_frames - 1)
    t_end = t_now + dt * num_frames_after
    t_steps = pd.date_range(t_start, t_end, freq='96min').values.astype('datetime64[s]') #otherwise its [ns]. Will convert to it in xarray anyway.
    return t_steps
    
    
def plot_heatmaps_info(imgs, algorithms, info,
                       zmin, zmax, color_continuous_scale,
                       animation_frame=None,
                       **kwargs
                      ):
    if animation_frame is not None:
        # Use the timestamps in info
        dims = ('Algorithm', 'Time', '\n', '')
        # h,w can't be both ''. They are used to index
        # specifying labels in px.imshow doesn't work
        coords = {'Algorithm': algorithms, 'Time': info['t_end']}
    else:
        dims = ('Algorithm', '\n', '')
        coords = {'Algorithm': algorithms}
    imgs = xr.DataArray(imgs, dims=dims, coords=coords)
    fig = plot_heatmaps(imgs, zmin, zmax, color_continuous_scale,
                        animation_frame=animation_frame, # 1 or None
                        facet_col=0,
                        **kwargs
                       )
    
    # how prob or prob0,1... are defined in df_box needs to be consistent.
    # we can implement a heuristic here
    #val_split = info['model_query'].split('/')[-3]
    #prob = info['prob' + val_split]
    #prob = info['prob'].iloc[0]
    #label = info['label'].iloc[0]
    #fig = add_label(fig, label, label_color="Red" if label else "Green")
    #fig = add_pred(fig, prob, pred_color="Red" if prob > 0.5 else "Green")
    
    # I don't know how to add annotation and shape to each frame
    # Each frame's layout.annotations is an emtpy tuple
    # Tried:
    # fig.layout.annotations[-2:]
    # fig.frames[1].layout.annotations = (go.layout.Annotation({
    #     'bgcolor': 'Blue', # change from Green
    #     'font': {'color': 'White', 'size': 12},
    #     'showarrow': False,
    #     'text': 'Prob 0.0145',
    #     'x': 1,
    #     'xanchor': 'right',
    #     'xref': 'x domain',
    #     'y': 0,
    #     'yanchor': 'bottom',
    #     'yref': 'y domain'
    # }),)
    # shape is also one attribute of layout
    
    # Facet subtitle implemented as annotations. This doesn't work:
    # for i in range(len(fig.frames)):
    #     fig.frames[i].layout.update(title=info['prob'].iloc[i])
    
    fig.update_layout(
        #dragmode='drawopenpath',
        newshape=dict(line_color='cyan'),
        #height=600,
        # width=300,
    )
    #fig.show(
    #    config={'modeBarButtonsToAdd':['drawopenpath', 'eraseshape']}
    #)
    return fig


def plot_heatmaps_contour(images, attribution_maps,
                          sigma=1,
                          linewidth=1.5,
                          figsize=(8,4),
                          headers=None,
                          info=None,
                          sunpy_map=False,
                          outlier_perc=2,
                         ):
    """
    Plot images with attribution map contours
    """
    figs = {}
    vmax = np.percentile(np.absolute(images), 99)
    #vmax = 457 # HARP 5982
    vmax = 500
    for algorithm, heatmap in attribution_maps.items():
        vmax_level = np.percentile(np.absolute(heatmap), 100-outlier_perc) # perceptive max: 0.0115
        figs[algorithm] = []
        for t in range(len(images)):
            mask = heatmap[t][0,0]
            mask_smoothed = gaussian_filter(mask, sigma=sigma)

            fig = plt.figure(figsize=figsize)
            if sunpy_map == False:
                ax = fig.add_subplot(121)
                ax.imshow(images[0][0,0], vmin=-vmax, vmax=vmax, cmap='gray')
                if info is not None:
                    ax.set_title((info['prefix'].iloc[0] + ' ' +
                                  str(info['arpnum'].iloc[0]) + '   ' +
                                  info['t_end'].iloc[0]))
            else:
                assert headers is not None
                magnetogram = sunpy.map.Map(images[0][0,0], dict(headers.iloc[0]))
                ax = fig.add_subplot(121, projection=magnetogram)
                magnetogram.plot(axes=ax, vmin=-vmax, vmax=vmax)
                magnetogram.draw_grid()
            # Can't draw contours anymore if I plot sunpy map on this axes.
            # Was hoping adding a new axis other than WCSAxes would fix the bug
            # ax = fig.add_subplot(121)
            ax.contour(mask_smoothed,
                       vmin=-vmax_level, vmax=vmax_level,
                       linewidths=linewidth,
                       levels=np.array([-vmax_level, vmax_level]),
                       cmap='bwr')
            #ax.axis('off') # Error if I use sunpy Map and don't turn off axis
            # Do not turn off axis. At least give the readers the pixels

            if sunpy_map == False:
                ax = fig.add_subplot(122)
                ax.imshow(images[t][0,0], vmin=-vmax, vmax=vmax, cmap='gray')
                if info is not None:
                    ax.set_title((info['prefix'].iloc[t] + ' ' +
                                  str(info['arpnum'].iloc[t]) + '   ' +
                                  info['t_end'].iloc[t]))
            else:
                assert headers is not None
                magnetogram = sunpy.map.Map(images[t][0,0], dict(headers.iloc[t]))
                ax = fig.add_subplot(122, projection=magnetogram)
                magnetogram.plot(axes=ax, vmin=-vmax, vmax=vmax)
                magnetogram.draw_grid()
            #ax = fig.add_subplot(122)
            ax.contour(mask_smoothed,
                       vmin=-vmax_level,
                       vmax=vmax_level,
                       levels=np.array([-vmax_level, vmax_level]),
                       linewidths=linewidth,
                       cmap='bwr')
            #ax.axis('off')
            fig.tight_layout()
            figs[algorithm].append(fig)
    return figs


def draw_contour(df):
    arpnum, t_end = df['arpnum'].iloc[0], df['t_end'].iloc[0]
    savedir = f'contours/{arpnum}/{t_end}'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    algorithm = 'DeepLift'# 'IntegratedGradients' #
    
    heatmaps = get_heatmaps_from_df(df, ['Original', algorithm], baseline_type='first')
    heatmaps = resize_heatmaps(heatmaps, df)
    headers = pd.concat([load_data_and_header(df['prefix'].iloc[i],
                                              df['arpnum'].iloc[i],
                                              pd.to_datetime(df['t_end'].iloc[i]),
                                              only_header=True)
                         for i in range(len(df))])
    figs = plot_heatmaps_contour(
        heatmaps['Original'],
        {algorithm: heatmaps[algorithm]},
        sigma=3, # 5982: 2
        figsize=(12, 3),
        linewidth=2,
        info=df,
        outlier_perc=1, # 5982: 1.1
    )
    
    # Save figure
    for Algorithm in figs:
        for t, fig in enumerate(figs[Algorithm]):
            fig.savefig(os.path.join(savedir, f'{t}.png'), dpi=300)

    # Save animation
    filenames = sorted(list(glob(savedir + '/*.png')),
                       key=lambda path: int(path.split('/')[-1].replace('.png', '')))
    images = list(map(lambda filename: imageio.imread(filename), filenames))
    imageio.mimsave(os.path.join(savedir, 'contour_movie.gif'), images, duration = 0.5) # modify duration as needed
    
    # Time series
    val_split = df['model_query'].iloc[0].split('/')[-3]
    df_flares = goes_df.loc[
        (goes_df['noaa_active_region'].isin(harpnum_to_noaa[arpnum])) &
        (goes_df['start_time'] >= df['t_end'].min()) &
        (goes_df['start_time'] - timedelta(hours=24)<= df['t_end'].max())]
    fig = plot_time_series(arpnum, df, df_flares, prob_columns=[f'prob_CNN_{val_split}']) # for val_split in range(5)])
    #fig.show()
    fig.update_layout(
        height=200,
        width=900,
        margin=go.layout.Margin(
            l=0, #left margin
            r=0, #right margin
            b=0, #bottom margin
            t=30  #top margin
        )
    )
    #fig.show(renderer="png")
    fig.write_image(os.path.join(savedir, 'time_series.pdf'))


def draw_attribution(df, algorithms):
    df = df.reset_index(drop=True)
    zmin, zmax = None, None
    color_continuous_scale = [
        px.colors.sequential.gray,
        *([px.colors.diverging.RdBu_r] * (len(algorithms)-1)), #balance # white for 0.
    ]
    
    heatmaps = get_heatmaps_from_df(df, algorithms, baseline_type='first')

    heatmaps = resize_heatmaps(heatmaps, df)

    #### Movie
    # # when df is a continuous time series for an active region
    # # concatenate all heatmaps along the time dimension
    # imgs = np.array([np.concatenate(heatmaps[a], axis=1)[0] for a in heatmaps])
    # algorithms = list(heatmaps.keys())
    # fig = plot_heatmaps_info(imgs, algorithms, df,
                             # zmin, zmax, color_continuous_scale,
                             # animation_frame=1, facet_col_wrap=2)
    # fig.show(config={'modeBarButtonsToAdd':['drawopenpath', 'eraseshape']})
    # #fig.write_html('captum_movie_first.html')
    
    #### Static (Practically, only the last frame is useful)
    g = lambda m: m[0, 0, :, :]
    # for i in df.index[[-1]]:
    i = df.index[-1]
    imgs = np.array([g(heatmaps[a][i]) for a in heatmaps])
    algorithms = list(heatmaps.keys())
    fig = plot_heatmaps_info(imgs, algorithms, df.iloc[i],
                             zmin, zmax, color_continuous_scale,
                             #facet_col_wrap=3,
                             animation_frame=None)
    kwargs = {f'{xy}axis{i}': {'showticklabels': False}
              for i in ['', *range(2, len(algorithms)+1)]
              for xy in ['x', 'y']}
    #height = 200
    width = 1500
    wh_ratio = len(algorithms) * 0.95 * (imgs.shape[2] / imgs.shape[1])
    # Width should be constant as we will put it in paper, on website, etc.
    fig.update_layout(
        margin={'l': 0, 'r': 0, 't': 20, 'b': 0}, # title seems to be out of margin
        height=width / wh_ratio,
        width=width,
        **kwargs,
    )
    # Do not show. Too big. 384 MB for two images
    #fig.show(config={'modeBarButtonsToAdd':['drawopenpath', 'eraseshape']})
    filename = 'contours/{}/{}/{}'.format(df.loc[i, 'arpnum'],
                                          df.loc[0, 't_end'],
                                          'last')
    fig.write_image(filename + '.png')
    fig.write_image(filename + '.pdf')
    return fig


def plot_heatmaps_overlay(heatmaps):
    images = heatmaps['Original']
    attrs = heatmaps['GuidedBackprop']
    outlier_perc = 0.2
    
    figs = []
    vmax_image = np.percentile(np.absolute(images), 99)
    vmax_attr = np.percentile(np.abs(np.ravel(attrs)), 100-outlier_perc)
    for t in range(len(images)):
        #if t != 8:
        #    continue
        image = images[t][0,0]
        attr = attrs[t][0,0]

        fig, _ = visualize_image_attr(
            attr,
            image,
            method='blended_heatmap', #'masked_image', #'blended_heatmap',
            sign='absolute', #'all', #'absolute',
            thresh_image=vmax_image,
            thresh_attr=vmax_attr,
        )
        figs.append(fig)
    return figs

        
def visualize_image_attr(
        attr,
        image,
        method='blended_heatmap',
        sign='all',
        thresh_image=None,
        thresh_attr=None,
        cmap_image=None,
        cmap_attr=None,
        alpha_overlay=0.4,
        show_colorbar=True,
    ):
    """
    Adapted from Captum. Not finished.
    
    Args:
        attr (ndarray): Attribution map.
        image (ndarray): Original image.
        method (str): Options are
            - 'blended_heatmap'
            - 'alpha_scaling'. Incompatible with sign='all'.
            - 'masked_image'. Incompatible with sign='all'.
        sign (str): Options are
            - 'all'. Incompatible with method='alpha_scaling' or 'masked_image'.
            - 'absolute'
            - 'positive'
        cmap_attr (str): Defaults to 'bwr' for all sign, 'Blues' for absolute
            sign, and 'Greens' for positive sign.
    """
    # begin of function
    assert image.ndim <= 3 and attr.ndim <= 3
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    if attr.ndim == 3:
        attr = np.mean(attr, axis=2)
        
    vmax_image = thresh_image
    vmin_image = -thresh_image if vmax_image else None
    
    vmax_attr = thresh_attr
    vmin_attr = -vmax_attr if sign == 'all' else 0
    
    cmap_image = cmap_image or 'gray' #'coolwarm'

    if cmap_attr is None:
        if sign == 'all':
            cmap_attr = 'bwr'
        elif sign == 'absolute':
            cmap_attr = 'hot' #'Blues' # 'hot'
        elif sign == 'positive':
            cmap_attr = 'Greens'
        else:
            raise

    if sign == 'absolute':
        attr = np.abs(attr)
    elif sign == 'positive':
        attr = attr * (attr > 0)
    norm_image = Normalize(vmin=vmin_image, vmax=vmax_image, clip=True)
    norm_attr = Normalize(vmin=vmin_attr, vmax=vmax_attr, clip=True)

    heatmap = None
    fig, ax = plt.subplots(figsize=(6,6))
    if method == 'blended_heatmap':
        ax.imshow(image, norm=norm_image, cmap=cmap_image)
        heatmap = ax.imshow(attr, alpha=alpha_overlay, norm=norm_attr, cmap=cmap_attr)
    elif method == 'alpha_scaling':
        assert sign != 'all'
        _cmap = plt.get_cmap(cmap_image)
        _image = _cmap(norm_image(image))
        _image[..., -1] = norm_attr(attr)
        ax.imshow(_image)
    elif method == 'masked_image':
        assert sign != 'all'
        _attr = norm_attr(attr)
        ax.imshow(image * attr, cmap=cmap_image)
        
    if show_colorbar:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        axis_separator = make_axes_locatable(ax)
        ax_colorbar = axis_separator.append_axes("right", size="5%", pad=0.1)
        if heatmap:
            fig.colorbar(heatmap, orientation="vertical", cax=ax_colorbar)
        else:
            ax_colorbar.axis('off')
    
    ax.axis('off')
    return fig, ax


def plot_heatmaps(imgs, zmin, zmax, color_continuous_scale, animation_frame=None, facet_col=None, **kwargs):
    """
    Args:
        imgs: np.ndarray of shape (N, H, W)
    """
    ## Solution 0: matplotlib
    #norm = Normalize(vmin=-0.1, vmax=0.1, clip=True)
    #f = lambda m: norm(m[0].transpose(1,2,0))
    #plt.figure(figsize=(12, 4))
    #plt.subplot(131); plt.imshow(f(heatmaps[0]))
    #plt.subplot(132); plt.imshow(f(heatmaps[1]))
    #plt.subplot(133); plt.imshow(f(heatmaps[1]-heatmaps[0]))
    #plt.show()
    
    ## Solution 1: px.imshow with facets
    # * zmin, zmax has to be scalar. I have modified plotly to support lists when binary_string=True (go.Heatmap is used). But I don't know
    # * binary_string = True will ignore color_continuous_scale.
    fig = px.imshow(
        imgs,
        zmin=zmin,
        zmax=zmax,
        animation_frame=animation_frame,
        facet_col=facet_col,
        facet_col_spacing=0.05, # default to 0.02
        #facet_col_wrap=3,
        binary_string=False,
        color_continuous_scale=color_continuous_scale,
        aspect='equal', # None means 'equal' for ndarray but 'auto' for xarray
        #labels={'h': '', 'w': ''}, # doesn't work
        **kwargs
    )

    ## Solution 2: subplots
    # * Unified colorbar has overlapping ticks, whereas separate colorbars require manually specifying positions. [link](https://community.plotly.com/t/subplots-of-two-heatmaps-overlapping-text-colourbar/38587/3)
    # * Can't specify colorscale does not work
    #fig = make_subplots(rows=1, cols=len(imgs),
    #                    shared_xaxes=True,
    #                    shared_yaxes=True,
    #                   )
    #for i, img in enumerate(imgs):
    #    trace = go.Heatmap(
    #        z=img,
    #        #coloraxis='coloraxis', # uncomment for unified colorscale
    #    )
    #    fig.add_trace(trace, 1, i+1)
    #kwargs = {}
    #kwargs.update({
    #    f'xaxis{i}': {
    #        'scaleanchor': f'y{i}',
    #        'constrain': 'domain',
    #    }
    #    for i in ['', *range(2, len(imgs)+1)]
    #})
    #kwargs.update({
    #    f'yaxis{i}': {
    #        'autorange': 'reversed',
    #        'constrain': 'domain',
    #    }
    #    for i in ['', *range(2, len(imgs)+1)]
    #})
    #from _plotly_utils.basevalidators import ColorscaleValidator
    #colorscale_validator = ColorscaleValidator("colorscale", "imshow")
    #colorscale = colorscale_validator.validate_coerce(px.colors.diverging.RdBu)
    #kwargs.update({
    #    f'coloraxis{i}': {
    #        'colorscale': colorscale, #px.colors.diverging.RdBu, #'gray', #colorscale[j], #none of them works
    #        'cmin': zmin[j],
    #        'cmid': 0, # no effect
    #        'cmax': zmax[j],
    #    }
    #    for j, i in enumerate(['', *range(2, len(imgs)+1)])
    #})
    #fig.update_layout(
    #    #coloraxis={'colorscale': 'RdBu_r'},  # uncomment for unified colorscale
    #    **kwargs
    #)
    ##fig.update_xaxes(matches='x') # don't need this if we already specified in make_subplots
    
    return fig


def add_label(fig, label, label_color=None):
    fig.add_shape(
        type='rect',
        x0=0, x1=127, y0=0, y1=127,
        line=dict(color=label_color, width=3),
        row=0, col=1
    )
    fig.add_annotation(
        text='Label {}'.format('+' if label else '-'),
        bgcolor=label_color,
        xref="x domain", yref="y domain",
        x=0, y=1,
        xanchor='left', yanchor='top',
        font=dict(color='White', size=12),
        showarrow=False,
        row=0, col=1,
    )
    return fig


def add_pred(fig, prob, pred_color=None):
    fig.add_annotation(
        text=f'Prob {prob:.4f}',
        bgcolor=pred_color,
        xref="x domain", yref="y domain",
        x=1, y=0,
        xanchor='right', yanchor='bottom',
        font=dict(color='White', size=12),
        showarrow=False,
        row=0, col=1,
    )
    return fig


def get_fits_data_filepath(prefix, arpnum, time):
    if prefix == 'HARP':
        series = 'hmi.sharp_cea_720s'
        data_dir = '/data2/SHARP/image/'
    elif prefix == 'TARP':
        series = 'mdi.smarp_cea_96m'
        data_dir = '/data2/SMARP/image/'
    else:
        raise

    t = time.strftime('%Y%m%d_%H%M%S_TAI')
    filename = f'{series}.{arpnum}.{t}.magnetogram.fits'
    filepath = os.path.join(data_dir, f'{arpnum:06d}', filename)
    return filepath


def get_fits_header_filepath(prefix, arpnum):
    if prefix == 'HARP':
        database = 'SHARP'
    elif prefix == 'TARP':
        database = 'SMARP'
    else:
        raise

    filepath = f'/data2/{database}/header/{prefix}{arpnum:06d}_ATTRS.csv'
    return filepath


def load_data_and_header(prefix, arpnum, time, only_header=False):
    """
    Args:
        prefix (str):
        arpnum (int):
        time (datetime):
        only_header (bool):
    To show the data:
    ```python
    sunpy.map.Map(data, dict(header.iloc[0])).peek()
    ```
    """
    from astropy.io import fits
    
    if not only_header:
        filepath = get_fits_data_filepath(prefix, arpnum, time)
        data = fits.open(filepath)[1].data
    
    t_rec = time.strftime('%Y.%m.%d_%H:%M:%S_TAI')
    filepath = get_fits_header_filepath(prefix, arpnum)
    header = pd.read_csv(filepath)
    header = header[header['T_REC'] == t_rec]

    if only_header:
        return header
    else:
        return data, header
