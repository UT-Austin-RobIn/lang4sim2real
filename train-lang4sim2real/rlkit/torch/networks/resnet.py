from typing import Type, Callable, Union, List, Optional, Dict

import numpy as np
import torch
from torch import (nn, Tensor)
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.models.resnet import (
    Bottleneck,
    conv1x1,
    conv3x3,
)

from rlkit.torch.networks.film import FiLMBlockSequence
from rlkit.util.pythonplusplus import identity


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        film_blocks: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample
        # the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.film_blocks = film_blocks

    def forward(self, input_dict: Dict) -> Dict:
        x = input_dict['x']
        film_inputs = input_dict['film_inputs']
        # list where ith elem = emb for ith film block

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.film_blocks is not None:
            out = self.film_blocks(out, film_inputs)

        out += identity
        out = self.relu(out)

        out_dict = {'x': out, 'film_inputs': film_inputs}
        return out_dict


def make_fc_layers(fc_layers: int, input_size: int):
    fc_layer_ops = []
    for i in range(len(fc_layers)):
        if i == 0:
            curr_layer = nn.Linear(input_size, fc_layers[0])
        else:
            curr_layer = nn.Linear(fc_layers[i-1], fc_layers[i])
        fc_layer_ops.append(curr_layer)
    return nn.Sequential(*fc_layer_ops)


class ResNet(nn.Module):
    # Adapted from
    # http://pytorch.org/vision/master/_modules/torchvision/models/resnet.html#resnet18
    def __init__(
        self,
        fc_layers: List[int],
        output_activation=identity,
        block: Type[Union[BasicBlock, Bottleneck]] = BasicBlock,
        # BasicBlock for 18, 34; Bottleneck for resnet-50,101,152
        layers: List[int] = [2, 2, 2, 2],  # AKA block sizes
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        conv_strides: List[int] = [2, 1, 2, 2, 2],
        maxpool_stride: int = 2,
        num_channels: List[int] = [64, 128, 256, 512],
        film_emb_dim_list: List[int] = [],  # [] == don't use film.
        num_film_inputs: int = 0,  # number of separate embeddings.
        # There will be `num_film_inputs` film blocks for each BasicBlock.
        film_hidden_sizes: List[int] = [],
        film_hidden_activation="identity",
        use_film_attn: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.output_activation = output_activation
        assert len(conv_strides) == 5

        # saving for phase 1 / 2 reference:
        self.layers = layers
        self.num_channels = num_channels
        self.conv_strides = conv_strides

        self.inplanes = num_channels[0]
        self.num_channels = num_channels
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=conv_strides[0],
            padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=maxpool_stride, padding=1)

        self.film_emb_dim_list = film_emb_dim_list
        self.num_film_inputs = num_film_inputs
        self.film_hidden_sizes = film_hidden_sizes
        self.film_hidden_activation = film_hidden_activation
        if use_film_attn:
            self.num_film_blocks_per_seq = 1
        else:
            self.num_film_blocks_per_seq = self.num_film_inputs

        self.layer1 = self._make_layer(
            block, num_channels[0], layers[0], stride=conv_strides[1])
        self.layer2 = self._make_layer(
            block, num_channels[1], layers[1], stride=conv_strides[2],
            dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block, num_channels[2], layers[2], stride=conv_strides[3],
            dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            block, num_channels[3], layers[3], stride=conv_strides[4],
            dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_head_input_size = num_channels[-1] * block.expansion
        self.fc_layer_ops = make_fc_layers(fc_layers, self.fc_head_input_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual
        # block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        film_blocks = FiLMBlockSequence(
            self.film_emb_dim_list, self.num_film_blocks_per_seq, planes,
            self.film_hidden_sizes, self.film_hidden_activation)
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups,
                self.base_width, previous_dilation, norm_layer,
                film_blocks=film_blocks,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            film_blocks = FiLMBlockSequence(
                self.film_emb_dim_list, self.num_film_blocks_per_seq, planes,
                self.film_hidden_sizes, self.film_hidden_activation)
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    film_blocks=film_blocks,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(
            self, x: Tensor, film_inputs: Optional[Tensor] = None,
            output_stage="") -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        out_dict = self.layer1(dict(x=x, film_inputs=film_inputs))
        if output_stage == "layer1":
            return out_dict['x']
        out_dict = self.layer2(out_dict)
        if output_stage == "layer2":
            return out_dict['x']
        out_dict = self.layer3(out_dict)
        out_dict = self.layer4(out_dict)

        x = out_dict['x']
        conv_channels = x

        if output_stage == "conv_channels":
            return conv_channels

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if output_stage == "last_activations":
            return x

        for fc in self.fc_layer_ops:
            x = fc(x)

        x = self.output_activation(x)

        if output_stage == "last-and-conv_channels":
            embs_out_dict = {
                "last": x,
                "conv_channels": conv_channels,
            }
            return embs_out_dict

        return x

    def forward(
            self, x: Tensor, film_inputs: List[Tensor] = [],
            output_stage="") -> Tensor:
        return self._forward_impl(x, film_inputs, output_stage)
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_last_layer(self):
        for param in self.layer4.parameters():
            param.requires_grad = True

    def unfreeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = True

    def change_film_block_requires_grad(self, requires_grad):
        for i, layer in enumerate(
                [self.layer1, self.layer2, self.layer3, self.layer4]):
            for basic_block in layer:
                for name, param in basic_block.film_blocks.named_parameters():
                    param.requires_grad = requires_grad

    def freeze_film_blocks(self):
        self.change_film_block_requires_grad(requires_grad=False)

    def unfreeze_film_blocks(self):
        self.change_film_block_requires_grad(requires_grad=True)

    def add_film_blocks(
            self,
            film_emb_dim_list: List[int] = [],  # [] == don't use film.
            num_film_inputs: int = 0,  # number of separate embeddings.
            film_hidden_sizes: List[int] = [],
            film_hidden_activation="identity"):
        for i, layer in enumerate(
                [self.layer1, self.layer2, self.layer3, self.layer4]):
            for basic_block in layer:
                basic_block.film_blocks = FiLMBlockSequence(
                    film_emb_dim_list, num_film_inputs, self.num_channels[i],
                    film_hidden_sizes, film_hidden_activation)


class ResNetMultiHead(nn.Module):
    """
    Resnet wrapper with multiple fc heads
    """
    def __init__(
            self,
            fc_dims_by_head: List[List[int]],
            output_activations: List[str],
            *args,
            **kwargs):
        assert kwargs['fc_layers'] == []
        super().__init__()
        self.resnet = ResNet(*args, **kwargs)
        self.fc_heads = nn.ModuleList([
            make_fc_layers(fc_dims, self.resnet.fc_head_input_size)
            for fc_dims in fc_dims_by_head])
        self.output_activations = output_activations
        assert len(self.fc_heads) == len(self.output_activations)
        self.num_heads = len(self.fc_heads)

    def forward(self, x: Tensor, film_inputs: List[Tensor] = []):
        img_emb = self.resnet(x, film_inputs, output_stage="last_activations")
        outs_by_head = []
        for i in range(self.num_heads):
            out = self.fc_heads[i](img_emb)
            out = self.output_activations[i](out)
            outs_by_head.append(out)
        return outs_by_head


class SpatialSoftmax(nn.Module):
    """Copied from here:
    https://github.com/rail-berkeley/railrl-private/blob/df4d24c328d7bdf3393043614981a52da5b01448/rlkit/torch/networks/resnet.py#L356
    """
    def __init__(
            self, height, width, channel, temperature=None,
            data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = nn.Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.height),
            np.linspace(-1., 1., self.width)
        )
        self.pos_x = torch.from_numpy(
            pos_x.reshape(self.height * self.width)).float()
        self.pos_y = torch.from_numpy(
            pos_y.reshape(self.height * self.width)).float()
        self.register_buffer('_pos_x', self.pos_x)
        self.register_buffer('_pos_y', self.pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(
                -1, self.height * self.width)
        else:
            feature = feature.reshape(-1, self.height * self.width)
            # feature = feature.view(-1, self.height * self.width)
        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(
            Variable(self._pos_x) * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(
            Variable(self._pos_y) * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel * 2)

        return feature_keypoints
