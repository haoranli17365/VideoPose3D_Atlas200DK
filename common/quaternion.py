# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
#import torch

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    # might cause error since we changed troch.cross to np.cross!
    uv = np.cross(qvec, v, axis=len(q.shape)-1)
    uuv = np.cross(qvec, uv, axis=len(q.shape)-1)
    return (v + 2 * (q[..., :1] * uv + uuv))
    
    
#def qinverse(q, inplace=False):
 #   # We assume the quaternion to be normalized
  #  if inplace:
   #     q[..., 1:] *= -1
    #    return q
   # else:
    #    w = q[..., :1]
     #   xyz = q[..., 1:]
      #  return torch.cat((w, -xyz), dim=len(q.shape)-1)
