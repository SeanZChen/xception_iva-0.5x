# Copyright (c) 2017-present, yszhu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""darknet19 from https://pjreddie.com/darknet/imagenet/"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
from maskrcnn_benchmark.layers import FrozenBatchNorm2d
from maskrcnn_benchmark.utils.registry import Registry

_NORMALIZATION_MODULES = Registry({
    "FrozenBatchNorm2d": FrozenBatchNorm2d,
    "BatchNorm2d": nn.BatchNorm2d,
})

def add_stage(
    n,
    dim_in,
    dim_out1,
    dim_inner,
    normalization
):
    """Add a ResNet stage to the model by stacking n residual blocks."""
    blocks = []
    for i in range(n):
        blocks.append(Xception_block(
            dim_in,
            dim_out1,
            dim_inner,
            normalization = normalization
        ))
        dim_in = dim_in + dim_inner + dim_inner
    return nn.Sequential(*blocks)

class Xception_block(nn.Module):
    def __init__(self,
        dim_in,
        dim_out1,
        dim_inner,
        normalization = 'FrozenBatchNorm2d'
        ):
        super(Xception_block, self).__init__()
        norm_func = _NORMALIZATION_MODULES[normalization]

        self.branch1a = nn.Conv2d(dim_in, dim_out1, kernel_size=1, padding=0, stride=1, bias=False)
        self.branch1a_bn = norm_func(dim_out1)
        
        self.branch1b = nn.Conv2d(dim_out1, dim_inner, kernel_size=3, padding=1, stride=1, bias=False)
        self.branch1b_bn = norm_func(dim_inner)
        #
        self.branch2a = nn.Conv2d(dim_in, dim_out1, kernel_size=1, padding=0, stride=1, bias=False)
        self.branch2a_bn = norm_func(dim_out1)
        
        self.branch2b = nn.Conv2d(dim_out1, dim_inner, kernel_size=3, padding=1, stride=1, bias=False)
        self.branch2b_bn = norm_func(dim_inner)
        
        self.branch2c = nn.Conv2d(dim_inner, dim_inner, kernel_size=3, padding=1, stride=1, bias=False)
        self.branch2c_bn = norm_func(dim_inner)

    def forward(self, x):
        b1 = self.branch1a(x)
        b1 = self.branch1a_bn(b1)
        b1 = F.relu_(b1)
        b1 = self.branch1b(b1)
        b1 = self.branch1b_bn(b1)
        b1 = F.relu_(b1)

        b2 = self.branch2a(x)
        b2 = self.branch2a_bn(b2)
        b2 = F.relu_(b2)
        b2 = self.branch2b(b2)
        b2 = self.branch2b_bn(b2)
        b2 = F.relu_(b2)
        b2 = self.branch2c(b2)
        b2 = self.branch2c_bn(b2)
        b2 = F.relu_(b2)

        return torch.cat([x, b1, b2], dim = 1)
#Xception_iva
#['res_stage4_tb_bn', 'res_stage3_tb_bn', 'res_stage2_tb_bn', 'res_stage1_tb_bn']
#[1. / 32., 1. / 16., 1. / 8., 1. / 4.]

class Xception_iva(nn.Module):
    def __init__(self, block_counts = (1, 2, 3, 3), normalization = "FrozenBatchNorm2d"):
        super(Xception_iva, self).__init__()
        self.norm_func = _NORMALIZATION_MODULES[normalization]

        self.stem1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2, bias=False)
        self.res_stem1_bn = self.norm_func(16)

        self.stem2a = nn.Conv2d(16, 16, kernel_size=1, padding=0, stride=1, bias=False)
        self.res_stem2a_bn = self.norm_func(16)
        self.stem2b = nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=2, bias=False)
        self.res_stem2b_bn = self.norm_func(16)

        self.stem3 = nn.Conv2d(32, 16, kernel_size=1, padding=0, stride=1, bias=False)
        self.res_stem3_bn = self.norm_func(16)

        (n1, n2, n3, n4) = block_counts[:4]
        dim_inner = 16
        self.stage1 = add_stage(n1, 16, 8, 8, normalization = normalization)
        self.stage1_tb = nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1, bias=False)
        self.res_stage1_tb_bn = self.norm_func(32)

        self.stage2 = add_stage(n2, 32, 16, dim_inner, normalization = normalization)
        self.stage2_tb = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1, bias=False)
        self.res_stage2_tb_bn = self.norm_func(96)
    
        self.stage3 = add_stage(n3, 96, 32, dim_inner, normalization = normalization)
        self.stage3_tb = nn.Conv2d(192, 192, kernel_size=1, padding=0, stride=1, bias=False)
        self.res_stage3_tb_bn = self.norm_func(192)
          
        self.stage4 = add_stage(n4, 192, 64, dim_inner, normalization = normalization)
        self.stage4_tb = nn.Conv2d(288, 288, kernel_size=1, padding=0, stride=1, bias=False)
        self.res_stage4_tb_bn = self.norm_func(288)
    

    def forward(self, x):
        x = self.stem1(x)
        x = self.res_stem1_bn(x)
        x = F.relu_(x)

        p2 = self.stem2a(x)
        p2 = self.res_stem2a_bn(p2)
        p2 = F.relu_(p2)
        p2 = self.stem2b(p2)
        p2 = self.res_stem2b_bn(p2)
        p2 = F.relu_(p2)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.cat([x, p2], dim=1)

        x = self.stem3(x)
        x = self.res_stem3_bn(x)
        x = F.relu_(x)

        f1 = self.stage1(x)
        f1 = self.stage1_tb(f1)
        f1 = self.res_stage1_tb_bn(f1)
        f1 = F.relu_(f1) 
        f2 = F.avg_pool2d(f1, kernel_size=2, padding=0, stride=2)

        f2 = self.stage2(f2)
        f2 = self.stage2_tb(f2)
        f2 = self.res_stage2_tb_bn(f2)
        f2 = F.relu_(f2)
        f3 = F.avg_pool2d(f2, kernel_size=2, padding=0, stride=2)
        
        f3 = self.stage3(f3)
        f3 = self.stage3_tb(f3)
        f3 = self.res_stage3_tb_bn(f3)
        f3 = F.relu_(f3)
        f4 = F.avg_pool2d(f3, kernel_size=2, padding=0, stride=2)
        
        f4 = self.stage4(f4)
        f4 = self.stage4_tb(f4)
        f4 = self.res_stage4_tb_bn(f4)
        f4 = F.relu_(f4)

        return [f1, f2, f3, f4]