#################################################################################
# Copyright (c) 2023-2026, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#################################################################################

"""Image classification models."""

from ..utils import py_utils
from .base import GenericModelWithSpec

class CNN_LENET5(GenericModelWithSpec):
    def __init__(self, config, input_features=(28,28), variables=1, num_classes=10):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_classes=num_classes)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_classes)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0':dict(type='BatchNormLayer', num_features=self.variables)}
        layers += {'1':dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(0,0))}
        layers += {'2':dict(type='MaxPoolLayer', kernel_size=(2,2), stride=(2,2), padding=(0,0))}
        layers += {'3':dict(type='ConvBNReLULayer', in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(0,0))}
        layers += {'4':dict(type='MaxPoolLayer', kernel_size=(2,2), stride=(2,2), padding=(0,0))}
        layers += {'5':dict(type='ReshapeLayer', ndim=2)}
        layers += {'6' :dict(type='LinearLayer', in_features=400, out_features=self.num_classes *12)}
        layers += {'7' :dict(type='ReluLayer')}
        layers += {'8' :dict(type='LinearLayer', in_features=self.num_classes*12, out_features=84)}
        layers += {'9' :dict(type='ReluLayer')}
        layers += {'10':dict(type='LinearLayer', in_features=84, out_features=self.num_classes)}
        model_spec = dict(model_spec=layers)
        return model_spec


class CNN_IMG_MOBILENETV1_58K_NPU(GenericModelWithSpec):
    """
    NPU-compliant MobileNetV1-inspired tiny model.
    - ~58K parameters
   
    Similarity to MobileNetV1:
    - Uses MobileNetV1-style depthwise separable blocks:
    Depthwise 3x3 convolution followed by Pointwise 1x1 convolution
    - Uses repeated DWCONV + PWCONV blocks for efficient spatial filtering and channel mixing
    - Uses progressive downsampling and global average pooling before the final classifier

    Architecture:
    BatchNorm -> Conv3x3/s2 ->[DWConv3x3 + PWConv1x1] ->[DWConv3x3/s2 + PWConv1x1] ->[DWConv3x3 + PWConv1x1] ->[DWConv3x3/s2 + PWConv1x1] ->[DWConv3x3 + PWConv1x1] ->[DWConv3x3/s2 + PWConv1x1] -> [DWConv3x3 + PWConv1x1] ->AdaptiveAvgPool -> FC
    """
    def __init__(self, config, input_features=(128, 128), variables=3, num_classes=10):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_classes=num_classes)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables,
                                   input_features=self.input_features, num_classes=self.num_classes)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        # Input: variables x 128 x 128
        layers += {'0' :dict(type='BatchNormLayer', num_features=self.variables)}
        # Stem conv: 128x128 -> 64x64
        layers += {'1' :dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=16,  kernel_size=(3,3), stride=(2,2), padding=(1,1))}
        # Depthwise separable block 1: 64x64 -> 64x64
        layers += {'2' :dict(type='ConvBNReLULayer', in_channels=16,  out_channels=16,  kernel_size=(3,3), stride=(1,1), padding=(1,1), groups=16)}
        layers += {'3' :dict(type='ConvBNReLULayer', in_channels=16,  out_channels=32,  kernel_size=(1,1), stride=(1,1), padding=(0,0))}
        # Depthwise separable block 2: 64x64 -> 32x32
        layers += {'4' :dict(type='ConvBNReLULayer', in_channels=32,  out_channels=32,  kernel_size=(3,3), stride=(2,2), padding=(1,1), groups=32)}
        layers += {'5' :dict(type='ConvBNReLULayer', in_channels=32,  out_channels=64,  kernel_size=(1,1), stride=(1,1), padding=(0,0))}
        # Depthwise separable block 3: 32x32 -> 32x32
        layers += {'6' :dict(type='ConvBNReLULayer', in_channels=64,  out_channels=64,  kernel_size=(3,3), stride=(1,1), padding=(1,1), groups=64)}
        layers += {'7' :dict(type='ConvBNReLULayer', in_channels=64,  out_channels=64,  kernel_size=(1,1), stride=(1,1), padding=(0,0))}
        # Depthwise separable block 4: 32x32 -> 16x16
        layers += {'8' :dict(type='ConvBNReLULayer', in_channels=64,  out_channels=64,  kernel_size=(3,3), stride=(2,2), padding=(1,1), groups=64)}
        layers += {'9' :dict(type='ConvBNReLULayer', in_channels=64,  out_channels=96,  kernel_size=(1,1), stride=(1,1), padding=(0,0))}
        # Depthwise separable block 5: 16x16 -> 16x16
        layers += {'10':dict(type='ConvBNReLULayer', in_channels=96,  out_channels=96,  kernel_size=(3,3), stride=(1,1), padding=(1,1), groups=96)}
        layers += {'11':dict(type='ConvBNReLULayer', in_channels=96,  out_channels=96,  kernel_size=(1,1), stride=(1,1), padding=(0,0))}
        # Depthwise separable block 6: 16x16 -> 8x8
        layers += {'12':dict(type='ConvBNReLULayer', in_channels=96,  out_channels=96,  kernel_size=(3,3), stride=(2,2), padding=(1,1), groups=96)}
        layers += {'13':dict(type='ConvBNReLULayer', in_channels=96,  out_channels=128, kernel_size=(1,1), stride=(1,1), padding=(0,0))}
        # Depthwise separable block 7: 8x8 -> 8x8
        layers += {'14':dict(type='ConvBNReLULayer', in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1), groups=128)}
        layers += {'15':dict(type='ConvBNReLULayer', in_channels=128, out_channels=128, kernel_size=(1,1), stride=(1,1), padding=(0,0))}
        # Global average pooling: 128 x 8 x 8 -> 128 x 1 x 1
        layers += {'16':dict(type='AdaptiveAvgPoolLayer', output_size=(1,1))}
        layers += {'17':dict(type='ReshapeLayer', ndim=2)}
        # Classifier: in_features=128
        layers += {'18':dict(type='LinearLayer', in_features=128, out_features=self.num_classes)}
        model_spec = dict(model_spec=layers)
        return model_spec

class CNN_IMG_MOBILENETV2_58K_NPU(GenericModelWithSpec):
    """
    NPU-compliant MobileNetV2-inspired probe model.
                                                                                                                            Closest possible sequential approximation of MobileNetV2
    using the current model_spec framework.                                                                               
    Included:
    - Inverted bottleneck blocks
    - Expansion PWCONV
    - Depthwise DWCONV
    - Projection PWCONV
    - Canonical MobileNetV2 channel schedule
    - AdaptiveAvgPool + FC head

    Not yet included:
    - Residual Add
    - ReLU6
    - Linear projection without ReLU
    """
    def __init__(self, config, input_features=(128, 128), variables=3, num_classes=4):
        super().__init__(config, input_features=input_features, variables=variables, num_classes=num_classes)
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec, variables=self.variables, input_features=self.input_features, num_classes=self.num_classes)

    def _add_inverted_bottleneck(self, layers, idx, in_ch, exp_ch, out_ch, stride):
        # 1x1 expansion
        layers += {f'{idx}a': dict(type='ConvBNReLULayer', in_channels=in_ch,  out_channels=exp_ch, kernel_size=(1,1), stride=(1,1),        padding=(0,0))}
        # 3x3 depthwise
        layers += {f'{idx}b': dict(type='ConvBNReLULayer', in_channels=exp_ch, out_channels=exp_ch, kernel_size=(3,3), stride=(stride,stride), padding=(1,1), groups=exp_ch)}
        # 1x1 projection
        # NOTE: Ideally this should be ConvBNLayer (linear projection). Using ConvBNReLULayer first to test compiler/NPU support.
        layers += {f'{idx}c': dict(type='ConvBNReLULayer', in_channels=exp_ch, out_channels=out_ch, kernel_size=(1,1), stride=(1,1),        padding=(0,0))}
        return out_ch

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        # Input: variables x 128 x 128
        layers += {'0' : dict(type='BatchNormLayer', num_features=self.variables)}
        # Stem conv: 128x128 -> 64x64
        layers += {'1' : dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(1,1))}
        in_ch = 32
        # MobileNetV2 canonical blocks: t=expansion ratio, c=output channels, n=repeats, s=stride
        cfg = [
            (1,  16, 1, 1),
            (6,  24, 2, 2),
            (6,  32, 3, 2),
            (6,  64, 4, 2),
            (6,  96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]
        idx = 2
        for t, c, n, s in cfg:
            for i in range(n):
                stride = s if i == 0 else 1
                exp_ch = in_ch * t
                in_ch = self._add_inverted_bottleneck(layers, idx, in_ch, exp_ch, c, stride)
                idx += 1
        # Final 1x1 conv
        layers += {f'{idx}' : dict(type='ConvBNReLULayer',      in_channels=in_ch, out_channels=1280, kernel_size=(1,1), stride=(1,1), padding=(0,0))}; idx += 1
        # Global average pooling -> 1280 x 1 x 1
        layers += {f'{idx}' : dict(type='AdaptiveAvgPoolLayer', output_size=(1,1))}; idx += 1
        layers += {f'{idx}' : dict(type='ReshapeLayer',         ndim=2)};           idx += 1
        # Classifier: in_features=1280
        layers += {f'{idx}' : dict(type='LinearLayer',          in_features=1280, out_features=self.num_classes)}
        return dict(model_spec=layers)

# Export all image classification models
__all__ = [
    'CNN_LENET5',
    'CNN_IMG_MOBILENETV1_58K_NPU',
    'CNN_IMG_MOBILENETV2_58K_NPU',
]
