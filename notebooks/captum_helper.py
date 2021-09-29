from datetime import datetime, timedelta
from ipdb import set_trace as breakpoint
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from captum.attr import IntegratedGradients, Saliency, DeepLift, GuidedBackprop, GuidedGradCam, Deconvolution, NoiseTunnel, LRP, LayerGradCam, LayerLRP

from arnet.utils import get_layer
from arnet.dataset import ActiveRegionDataModule, ActiveRegionDataset
from arnet.transforms import get_transform
from cotrain_helper import get_learner_by_query


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
                    target = 1
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
                       animation_frame=None):
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
                        facet_col=0)
    
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


def plot_heatmaps_contour(images, attribution_maps):
    """
    Plot images with attribution map contours
    """
    figs = {}
    vmax = np.percentile(np.absolute(images), 99)
    for algorithm, heatmap in attribution_maps.items():
        vmax_heatmap = np.percentile(np.absolute(heatmap), 99) # perceptive max: 0.0115
        vmax_level = vmax_heatmap * 0.8
        figs[algorithm] = []
        for t in range(len(images)):
            mask = heatmap[t][0,0]
            mask_smoothed = gaussian_filter(mask, sigma=1)

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
            ax[0].imshow(images[0][0,0], vmin=-vmax, vmax=vmax, cmap='gray')
            ax[0].contour(mask_smoothed,
                          vmin=-vmax_level, vmax=vmax_level,
                          linewidths=2,
                          levels=0.9 * np.array([-vmax_level, vmax_level]), # should be the most contrastive color so vmin and vmax
                          cmap='bwr')
            ax[0].axis('off')

            ax[1].imshow(images[t][0,0], vmin=-vmax, vmax=vmax, cmap='gray')
            ax[1].contour(mask_smoothed,
                          vmin=-vmax_level,
                          vmax=vmax_level,
                          levels=0.9 * np.array([-vmax_level, vmax_level]), # should be the most contrastive color so vmin and vmax
                          linewidths=2,
                          cmap='bwr')
            ax[1].axis('off')
            figs[algorithm].append(fig)
    return figs


def plot_heatmaps(imgs, zmin, zmax, color_continuous_scale, animation_frame=None, facet_col=None):
    """
    Args:
        imgs: np.ndarray of shape (N, H, W)
    """
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

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
