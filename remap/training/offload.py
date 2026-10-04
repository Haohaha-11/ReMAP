"""Keep autograd activations in host memory during long forward passes.

Like :class:`torch.autograd.graph.save_on_cpu`, but model parameters (and anything
sharing their storage) stay on the GPU and a tensor saved by several operations is
copied once.  Values are unchanged; only where they are kept differs.
"""
from __future__ import annotations

import weakref

import torch


class offload_activations(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, module: torch.nn.Module):
        resident = {t.untyped_storage().data_ptr() for t in list(module.parameters()) + list(module.buffers())}
        copies = {}

        def pack(tensor):
            if tensor.device.type == "cpu" or tensor.untyped_storage().data_ptr() in resident:
                return tensor
            key = id(tensor)
            entry = copies.get(key)
            if entry is not None and entry[0]() is tensor and entry[1] == tensor._version:
                return tensor.device, entry[2]
            host = tensor.detach().cpu()
            copies[key] = (weakref.ref(tensor), tensor._version, host)
            return tensor.device, host

        def unpack(packed):
            if isinstance(packed, torch.Tensor):
                return packed
            device, host = packed
            return host.to(device)

        super().__init__(pack, unpack)
