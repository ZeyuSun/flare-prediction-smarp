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
