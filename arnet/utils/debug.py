def plot(data, *args, **kwargs):
    import numpy as np
    import matplotlib; matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import torch
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif not isinstance(data, np.ndarray):
        data = np.array(data)

    if data.ndim == 1:
        plt.plot(data, *args, **kwargs)
        plt.legend()
    elif data.ndim == 2:
        plt.plot(data, *args, **kwargs)
        plt.colorbar()
    elif data.ndim == 3:
        if data.shape[2] not in {1, 3}:
            print("last dimension should be 1 or 3")
            raise
        plt.imshow(data)
        plt.colorbar()
    plt.show()