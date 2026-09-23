Fixed-Point Analysis
====================

``mFixedPointFinder`` supports both model types. Initial states are leaky ``x``
or Elman ``h``; inputs may be a shared vector ``[I]`` or one vector per candidate
``[N, I]``. ``find_fixed_points`` returns ``(unique_fps, all_fps)`` collections.
Optional positional region names select which parts of the state to optimize.

By default the objective is ``0.5 * ||F(state) - state||²``. For leaky models,
``optimize_h=True`` instead compares ``activation(x)`` with the next activity,
while still optimizing ``x``. In that mode ``xstar`` remains pre-activation and
``F_xstar`` contains next-step activity. The flag is ignored for Elman models.
The finder does not compute Jacobians; use :doc:`mrnntorch.analysis.linear`.

.. automodule:: mrnntorch.analysis.fp_finder
   :members:
   :show-inheritance:
