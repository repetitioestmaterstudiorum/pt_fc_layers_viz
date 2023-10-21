# pt_fc_layers_viz

`pt_fc_layers_viz` is a simple tool to visualize the architecture of each layer of a PyTorch fully connected neural network.

It shows

- Inputs
- Layers with nodes and activations
- Outputs
- Parameter values (weights and biases - optional)

## Usage

Install the package via pip: `pip install pt-fc-layers-viz`, then use it as follows in your code:

```python
from pt_fc_layers_viz import draw_pt_fc_layers

draw_pt_fc_layers(model, param_val=True, display_img=True, format='svg', rankdir='LR')
```

- `param_val` (default `True`): Show the parameter values (weights and biases).
- `display_img` (default `True`): Display the visualization inline in the notebook.
- `format` (default `'svg'`): The format to use for the visualization. Supported formats are 'svg', 'png', 'jpeg', and 'jpg'.
- `rankdir` (default `'LR'`): The direction of the graph. Use `'LR'` for left-to-right, and `'TB'` for top-to-bottom.

The function will automatically display the visualization (default setting). If you would like to save it as a file, you can call the `.view()` method on the return value (dot) of the function.

## Prerequisites

- graphviz (MacOS users can install it with `brew install graphviz`)
- torch
- IPython

## Example Visualizations

Neural net with layers:

    (0): Linear(in_features=2, out_features=4, bias=True)
    (1): ReLU()
    (2): Linear(in_features=4, out_features=2, bias=True)
    (3): ReLU()
    (4): Linear(in_features=2, out_features=1, bias=True)

With parameter values:
![Example Visualization](https://github.com/repetitioestmaterstudiorum/pt_fc_layers_viz/blob/main/assets/example-with-paramts.png?raw=true)

Without parameter values:
![Example Visualization](https://github.com/repetitioestmaterstudiorum/pt_fc_layers_viz/blob/main/assets/example-without-paramts.png?raw=true)
