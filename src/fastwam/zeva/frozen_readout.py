"""Evaluate a frozen base readout without keeping a second GPU backbone.

Only explicit joint main variants need this: their static retrieval head was
trained on the original foundation, not the updated action policy. CPU copies
keep both versions exact; the live module is restored even if readout fails.
This performs no optimization or deployment learning of model parameters.
"""
from contextlib import contextmanager
from threading import RLock

import torch


class FrozenReadoutSnapshot:
    def __init__(self, module):
        self.state = {name: value.detach().to("cpu", copy=True)
                      for name, value in module.state_dict().items()}
        self.lock = RLock()

    @contextmanager
    def activate(self, module):
        with self.lock, torch.no_grad():
            live = module.state_dict(keep_vars=True)
            if live.keys() != self.state.keys():
                raise ValueError("frozen retrieval backbone architecture changed")
            previous = {name: value.detach().to("cpu", copy=True) for name, value in live.items()}
            modes = [(child, child.training) for child in module.modules()]
            try:
                # copy_ streams each CPU tensor to its destination; do not
                # allocate a complete second model on the 32-GB deployment GPU.
                for name, value in live.items():
                    value.copy_(self.state[name])
                module.eval()
                yield
            finally:
                for name, value in live.items():
                    value.copy_(previous[name])
                for child, training in modes:
                    child.training = training
