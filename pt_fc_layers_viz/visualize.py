from graphviz import Digraph
from IPython.display import display, SVG

def get_pt_fc_layers(pt_fc_model):
    params_per_layer = {}

    # Add weights and biases
    for name, param in pt_fc_model.named_parameters():
        is_weight = name.endswith('weight')
        is_bias = name.endswith('bias')
        layer_id = name.rsplit('.', 1)[0]

        if layer_id not in params_per_layer:
            params_per_layer[layer_id] = {}

        if is_weight:
            params_per_layer[layer_id]['weights'] = param.data
            params_per_layer[layer_id]['input_size'] = param.shape[1]
            params_per_layer[layer_id]['output_size'] = param.shape[0]   
        elif is_bias:
            params_per_layer[layer_id]['biases'] = param.data

    layer_ids = list(params_per_layer.keys())
    named_modules = list(pt_fc_model.named_modules())

    # Add activation information
    for i, (name, param) in enumerate(named_modules):
        if i + 1 == len(named_modules):
            break # End of list is the next module
        
        if name in layer_ids:
            # Adding next module after this module
            layer_id = name.rsplit('.', 1)[1]
            _, param_next = named_modules[i + 1]
            params_per_layer[name]['activation'] = param_next.__class__.__name__
    del named_modules

    # Rectify layer names in params_per_layer from layers.0 to layer.1, etc.
    for i, layer_id in enumerate(layer_ids):
        params_per_layer[f'layer.{i}'] = params_per_layer.pop(layer_id)
    del layer_ids

    layers = list(params_per_layer.items())
    return layers


def dot_case_to_pascal_case(node_id):
    # Turns "layer.2.node.1" into "Layer 2, Node 1"
    node_id = node_id.replace('layer.', 'Layer ')
    node_id = node_id.replace('node.', ', Node ')
    node_id = node_id.replace('.', '')
    return node_id


def draw_pt_fc_layers(pt_fc_model, param_val=True):
    layers = get_pt_fc_layers(pt_fc_model)

    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    for i, (layer_id, layer_params) in enumerate(layers):
        input_size = layer_params['input_size']
        output_size = layer_params['output_size']
        
        # If this is the first layer, add the input nodes, in reverse order
        if i == 0:
            for i_is in range(input_size):
                input_node_id = f'input.{i_is}'
                dot.node(input_node_id, f'Input {i_is}')

        # Add a node per layer and a connection per input for each new node
        for i_os in range(output_size):
            node_id = f'{layer_id}.node.{i_os}'
            bias_str = f"Bias: {layer_params['biases'][i_os]:.4f}" if param_val else ''
            dot.node(node_id, f"{dot_case_to_pascal_case(node_id)}\n{layer_params.get('activation', 'No')} activation\n{bias_str}")
            
            # For each node in the previous layer, add a connection
            if i > 0 and i < len(layers):
                for i_is in range(input_size):
                    prev_layer_id = f'{layers[i - 1][0]}.node.{i_is}'
                    weight_str = f"Weight {i_os} {i_is}: {layer_params['weights'][i_os][i_is]:.4f}" if param_val else f"Weight {i_os} {i_is}"
                    dot.edge(prev_layer_id, node_id, label=weight_str)
            
            # If this is the last layer, add the output nodes
            if i == len(layers) - 1:
                output_node_id = f'output.{i_os}'
                dot.node(output_node_id, f'Output {i_os}')
                dot.edge(node_id, output_node_id)

        # If this is the first layer, add input edges
        if i == 0:
            for i_os in range(output_size):
                node_id = f'{layer_id}.node.{i_os}'
                for i_is in range(input_size):
                    input_node_id = f'input.{i_is}'
                    input_weight_str = f"Weight {i_os} {i_is}: {layer_params['weights'][i_os][i_is]:.4f}" if param_val else f"Weight {i_os} {i_is}"
                    dot.edge(input_node_id, node_id, label=input_weight_str)

    svg = dot.pipe(format='svg')
    display(SVG(svg))
    return dot
