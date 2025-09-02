from typing import Optional

from torch import nn
import torch as th


class PNNLayers(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        net_arch: list[int],
        activation_fn: type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
        with_bias: bool = True,
        history_modules: Optional[list["PNNLayers"]] = None,
        *args,
        **kwargs
    ):
        """
        Create a multi layer perceptron (MLP), which is
        a collection of fully-connected layers each followed by an activation function.

        :param input_dim: Dimension of the input vector
        :param output_dim: Dimension of the output (last layer, for instance, the number of actions)
        :param net_arch: Architecture of the neural net
           It represents the number of units per layer.
           The length of this list is the number of layers.
        :param activation_fn: The activation function
           to use after each layer.
        :param squash_output: Whether to squash the output using a Tanh
           activation function
        :param with_bias: If set to False, the layers will not learn an additive bias
        :param pre_linear_modules: List of nn.Module to add before the linear layers.
           These modules should maintain the input tensor dimension (e.g. BatchNorm).
           The number of input features is passed to the module's constructor.
           Compared to post_linear_modules, they are used before the output layer (output_dim > 0).
        :param post_linear_modules: List of nn.Module to add after the linear layers
           (and before the activation function). These modules should maintain the input
           tensor dimension (e.g. Dropout, LayerNorm). They are not used after the
           output layer (output_dim > 0). The number of input features is passed to
           the module's constructor.
        :return: The network module
        """
        super().__init__(*args, **kwargs)
        linears = []
        lateral_conns = []
        lateral_size = len(history_modules) if history_modules else 0

        if len(net_arch) > 0:
            linears.append(nn.Linear(input_dim, net_arch[0], bias=with_bias))
            lateral_conns.append(
                nn.ModuleList(
                    [
                        nn.Linear(input_dim, net_arch[0], bias=with_bias)
                        for _ in range(lateral_size)
                    ]
                )
            )

        for idx in range(len(net_arch) - 1):
            linears.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
            lateral_conns.append(
                nn.ModuleList(
                    [
                        nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias)
                        for _ in range(lateral_size)
                    ]
                )
            )

        if output_dim > 0:
            last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
            linears.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
            lateral_conns.append(
                nn.ModuleList(
                    [
                        nn.Linear(last_layer_dim, output_dim, bias=with_bias)
                        for _ in range(lateral_size)
                    ]
                )
            )

        self.linears = nn.ModuleList(linears)
        self.lateral_conns = nn.ModuleList(lateral_conns)
        self.activation_fn = activation_fn()
        self.lateral_size = lateral_size
        self.history_modules = history_modules if history_modules else []
        self.squash_output = squash_output

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: Input tensor
        :param history_modules: History modules
        :return: Output tensor after passing through the network
        """
        assert len(self.history_modules) == self.lateral_size

        lateral_outputs = [x for _ in range(self.lateral_size)]
        for i in range(len(self.linears)):
            layer = self.linears[i]
            x = layer(x)

            for j in range(self.lateral_size):
                x += self.lateral_conns[i][j](lateral_outputs[j])

                with th.no_grad():
                    history_module: "PNNLayers" = self.history_modules[j]  # type: ignore
                    history_layer = history_module.linears[i]
                    lateral_outputs[j] = history_layer(lateral_outputs[j]).detach()

            if self.activation_fn and i < len(self.linears) - 1:
                x = self.activation_fn(x)
                for j in range(self.lateral_size):
                    lateral_outputs[j] = self.activation_fn(lateral_outputs[j])

        if self.squash_output:
            x = nn.Tanh()(x)
        return x
