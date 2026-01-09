#################################################################################
# Copyright (c) 2023-2024, Texas Instruments
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

"""Forecasting models for time series data."""

from ..utils import py_utils
from .base import GenericModelWithSpec


class FC_CNN_TS_GEN_BASE_13K(GenericModelWithSpec):
    def __init__(self, config, input_features=1, variables=1, num_classes=1):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_classes=num_classes)
        # Generate and initialize the model specification
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec,
                                   variables=self.variables,
                                   input_features=self.input_features,
                                   num_classes=self.num_classes)

    def gen_model_spec(self):
        layers = py_utils.DictPlus()
        layers += {'0': dict(type='BatchNormLayer', num_features=self.variables)}
        layers += {'1':dict(type='ConvBNReLULayer', in_channels=self.variables, out_channels=32, kernel_size=(3,1), padding=(1,0), stride=(1,1))}
        layers += {'2':dict(type='ConvBNReLULayer', in_channels=32, out_channels=64, kernel_size=(3,1), padding=(1,0), stride=(1,1))}
        layers += {'3':dict(type='ReshapeLayer', ndim=2)}
        layers += {'4':dict(type='LinearLayer', in_features=None, out_features=self.num_classes)}

        model_spec = dict(model_spec=layers)
        return model_spec


class LSTM10_TS_GEN_BASE(GenericModelWithSpec):
    def __init__(self, config, input_features=1, variables=1, num_classes=1):
        super().__init__(config, input_features=input_features, variables=variables,
                        num_classes=num_classes)
        # Generate and initialize the model specification
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec,
                                   variables=self.variables,
                                   input_features=self.input_features,
                                   num_classes=self.num_classes)

    def gen_model_spec(self):
        # Define the layer specifications
        layers = py_utils.DictPlus()
        layers += {'0': dict(type='IdentityLayer', num_features=self.variables)}
        layers += {'1': dict(type='LSTMLayer', input_size=self.variables, hidden_size=10, batch_first=True, return_last_timestep=True)}
        layers += {'2': dict(type='LinearLayer', in_features=None, out_features=self.num_classes)}

        model_spec = dict(model_spec=layers)
        return model_spec


class LSTM8_TS_GEN_BASE(GenericModelWithSpec):
    def __init__(self, config, input_features=1, variables=1, num_classes=1):
        super().__init__(config, input_features=input_features, variables=variables,
                         num_classes=num_classes)
        # Generate and initialize the model specification
        self.model_spec = self.gen_model_spec()
        self._init_model_from_spec(model_spec=self.model_spec,
                                   variables=self.variables,
                                   input_features=self.input_features,
                                   num_classes=self.num_classes)

    def gen_model_spec(self):
        # Define the layer specifications
        layers = py_utils.DictPlus()
        layers += {'0': dict(type='IdentityLayer', num_features=self.variables)}
        layers += {'1': dict(type='LSTMLayer', input_size=self.variables, hidden_size=8, batch_first=True, return_last_timestep=True)}
        layers += {'2': dict(type='LinearLayer', in_features=None, out_features=self.num_classes)}


        model_spec = dict(model_spec=layers)
        return model_spec


# Export all forecasting models
__all__ = [
    'FC_CNN_TS_GEN_BASE_13K',
    'LSTM10_TS_GEN_BASE',
    'LSTM8_TS_GEN_BASE',
]
