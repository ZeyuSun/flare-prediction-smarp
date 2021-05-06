def get_flare_index(flares):
    """Daily SXR flare index (Abramenko 2005)"""
    weights = {
        'X': 100,
        'M': 10,
        'C': 1,
        'B': 0.1,
        'A': 0,
    }
    flare_index = 0
    for f in flares:
        if f == '':
            continue
        if f == 'C':
            continue
        flare_index += weights[f[0]] * float(f[1:])
    flare_index = round(flare_index, 1) # prevent numerical error
    return flare_index


def get_log_intensity(class_str):
    import numpy as np

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