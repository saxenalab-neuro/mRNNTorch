# mRNNTorch

[![Documentation](https://img.shields.io/badge/docs-Read%20the%20Docs-blue)](https://mrnntorch.readthedocs.io/en/main/)

mRNNTorch is a small Python package for building and analyzing multi-regional recurrent neural networks (mRNNs) in PyTorch. The `mRNN` module mirrors `torch.nn.RNN` usage while managing region-level connectivity, masks, and constraints so you can prototype quickly.

## Highlights
- Define recurrent and input regions in JSON, Python, or a mix of both
- Automatic assembly of block weight matrices and masks
- Optional sign constraints (Dale's Law) and spectral radius scaling
- Analysis helpers for fixed points, flow fields, and linearization

## Package layout
| Component | Description |
| --- | --- |
| `mrnntorch.mRNN` | Core mRNN module for building custom networks |
| `mrnntorch.region` | Region classes and connectivity primitives |
| `mrnntorch.analysis` | Analysis utilities operating on an mRNN instance |

## Installation
Install from source:

```bash
python -m pip install -e .
```

## Quick start

### Building mRNN

There are two ways to define an mRNN:
1. Pass a JSON config file for regions and connections.
2. Build regions and connections manually inside your model.

You can also mix both approaches, then finalize connectivity:

```python
import torch.nn as nn
from mrnntorch import mRNN

class MyMRNN(nn.Module):
    def __init__(self, config_path: str):
        super().__init__()
        self.mrnn = mRNN(config_path, config_finalize=False)

        # Add any extra recurrent connections not in the config
        regions = ["r1", "r2"]
        for src in regions:
            for dst in regions:
                self.mrnn.add_recurrent_connection(src, dst)

        # Required if you manually add regions/connections
        self.mrnn.finalize_connectivity()

    def forward(self, inp, x, h):
        return self.mrnn(inp, x, h)
```

### Analysis

Given a `MyMRNN` instance named `model` (for example, after training), generate a trajectory for analysis. These examples run on CPU with batch-first sequences and use the leaky network's pre-activation states `x`.

```python
import torch
from mrnntorch.analysis import mFixedPointFinder, mFlowFieldFinder, mLinearization

rnn = model.mrnn.cpu().eval()
rnn.batch_first = True
inputs = torch.randn(1, 100, rnn.W_inp.shape[1])
x0, h0 = rnn.batched_initial_condition(batch_size=1)
with torch.no_grad():
    states, activities = rnn(inputs, x0, h0, noise=False)
```

**Fixed points.** Sample initial guesses from the trajectory and search at a constant input. Check `xstar` to access the fixed points, and `F_xstar` to access the next state from said fixed point. Use `F_xstar` for leaky mRNNs when attempting to access network activations `h` instead of pre-activations `x`.

```python
fixed_input = torch.zeros(rnn.W_inp.shape[1])
fp_finder = mFixedPointFinder(rnn, lr_init=1e-2, max_iters=1000, verbose=False)
initial_states = fp_finder.sample_states(states, n_inits=32, noise_scale=0.01)
fixed_points, all_candidates = fp_finder.find_fixed_points(initial_states, fixed_input)
print(fixed_points.xstar)
print(fixed_points.F_xstar)
```

**Linearization.** Build a local linear model at each fixed point found above, using the same constant input. The Jacobian describes the discrete-time state update.

```python
linearizations = [
    mLinearization(rnn, fixed_input, state)
    for state in fixed_points.xstar
]
for lin in linearizations:
    eigenvalues = torch.complex(lin.jac_rec_real_eigs, lin.jac_rec_im_eigs)
    print("Spectral radius:", eigenvalues.abs().max().item())
    # lin.jacobian and lin.jacobian_inp contain state and input derivatives.
```

**Flow fields.** Fit a two-dimensional PCA plane to the trajectory, then compute a nonlinear flow field at the same constant input.

```python
flow_finder = mFlowFieldFinder(
    rnn,
    fit_states=states,
    num_points=20,
    x_offset=2,
    y_offset=2,
)
field = flow_finder.find_nonlinear_flow(states[:, -1:], fixed_input[None, None, :])[0]

import matplotlib.pyplot as plt

plt.quiver(
    field.grid[..., 0], field.grid[..., 1],
    field.x_vels, field.y_vels,
)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.show()
```

## Configuration files
The config format specifies recurrent/input regions and their connections. See `src/mrnntorch/examples/configurations/CBGTCL.json` for a concrete example.

## Requirements
Runtime dependencies:
- torch
- numpy
- matplotlib
- scikit-learn (analysis helpers)

## Status
This repository is still under active development; contributions are welcome.
