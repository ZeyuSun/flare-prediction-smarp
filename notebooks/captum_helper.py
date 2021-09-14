from ipdb import set_trace as breakpoint
import numpy as np
import torch
from skimage.transform import resize
from captum.attr import IntegratedGradients, Saliency, DeepLift, GuidedGradCam, NoiseTunnel, LayerGradCam, LayerLRP


def get_heatmap(algorithm, learner, input, target):
    """
    TODO: heatmaps should only have 1 channels
    """
    model = learner.model
    model.eval()
    model.zero_grad()    
    if algorithm == 'Original':
        heatmap = input
    elif algorithm == 'Saliency':
        saliency = Saliency(model)
        heatmap = saliency.attribute(
            input,
            target=target,
        )
    elif algorithm == 'IntegratedGradients':
        ig = IntegratedGradients(model)
        heatmap, delta = ig.attribute(
            input,
            target=target,
            baselines=input * 0,
            return_convergence_delta=True
        )
        print('Approximation delta: ', abs(delta))
    elif algorithm == 'NoiseTunnel':
        ig = IntegratedGradients(model)
        nt = AttrAlgo(ig)
        heatmap = nt.attribute(
            input,
            target=target,
            baselines=input * 0,
            nt_type='smoothgrad_sq',
            nt_samples=100,
            stdevs=0.2
        )
    elif algorithm == 'DeepLift':
        deeplift = DeepLift(model)
        heatmap = deeplift.attribute(
            input,
            target=target,
            baselines=input * 0,
        )
    elif algorithm == 'GuidedGradCam':
        guided_gradcam = GuidedGradCam(model, model.convs.conv4) #conv5)
        heatmap = guided_gradcam.attribute(
            input,
            target=target,
        )
    elif algorithm == 'LayerGradCam':
        layer_gradcam = LayerGradCam(model, model.convs.conv4)
        heatmap = layer_gradcam.attribute(
            input,
            target=target,
        )
        # up to 5D input is supported: (B, C, [D], [H], W)
        # the `size` argument does not include B and C
        heatmap = torch.nn.functional.interpolate(
            heatmap,
            size=input[0, 0].shape,
            mode='trilinear',
            align_corners=False,
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


def plot_heatmaps(imgs, zmin, zmax, color_continuous_scale):
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
        facet_col=0,
        facet_col_spacing=0.05, # default to 0.02
        facet_col_wrap=3,
        binary_string=False,
        color_continuous_scale=color_continuous_scale,
        aspect='equal', # None means 'equal' for ndarray but 'auto' for xarray
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


def add_label(fig, label, label_color):
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

    
def add_pred(fig, prob, pred_color):
    fig.add_shape(
        type='rect',
        x0=0, x1=127, y0=0, y1=127,
        line=dict(color=pred_color, width=3),
        row=0, col=1
    )
    fig.add_annotation(
        text=f'Prob {prob:.4f}',
        bgcolor=pred_color,
        xref="x domain", yref="y domain",
        x=1, y=1,
        xanchor='right', yanchor='top',
        font=dict(color='White', size=12),
        showarrow=False,
        row=0, col=1,
    )
    return fig