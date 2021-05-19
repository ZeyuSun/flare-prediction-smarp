def get_layer(model, layer_name):
    from operator import attrgetter

    layer = attrgetter(layer_name)(model)
    return layer


def register_single_activation(activations, model, layer_name):
    """
    Add an activation hook to the `layer_name` of the `model`
    """
    def get_activations(model, input, output):
        activations[layer_name] = output.clone().detach()

    layer = get_layer(model, layer_name)
    layer.register_forward_hook(get_activations)


def register_activations(model, layer_names):
    activations = {}
    for layer_name in layer_names:
        register_single_activation(activations, model, layer_name)
    return activations