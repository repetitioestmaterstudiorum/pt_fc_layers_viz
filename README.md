# pt_fc_layers_viz

`pt_fc_layers_viz` is a simple tool to visualize the architecture of each layer of a PyTorch fully connected neural network.

It shows

- Inputs
- Layers with nodes and activations
- Outputs
- Parameter values (weights and biases - optional)

## Usage

    ```python
    from pt_fc_layers_viz import pt_fc_layers_viz
    pt_fc_layers_viz(model, param_val=False|True)
    ```

The function will automatically display the visualization. If you would like to save it as svg file, you can call the `.view()` method on the return value of the function.

## Prerequisites

- graphviz (MacOS users can install it with `brew install graphviz`)
- torch
- IPython

## Example Visualizations

Neural net with layers:

- (0): Linear(in_features=2, out_features=2, bias=True)
- (1): Sigmoid()
- (2): Linear(in_features=2, out_features=1, bias=True)

Without parameter values:

![Example Visualization](https://github.com/repetitioestmaterstudiorum/pt_fc_layers_viz/blob/main/assets/visualization-xor-model.png?raw=true)

With parameter values:
![Example Visualization](https://github.com/repetitioestmaterstudiorum/pt_fc_layers_viz/blob/main/assets/visualization-xor-model-param-val.png?raw=true)
