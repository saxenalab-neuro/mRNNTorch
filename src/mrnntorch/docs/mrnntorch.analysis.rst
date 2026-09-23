Analysis API
============

The same analysis classes support both ``mRNN`` and ``ElmanmRNN``:

.. code-block:: python

   from mrnntorch.analysis import (
       mRNNAdapter, mFixedPointFinder, mFlowFieldFinder, mLinearization,
   )

Use pre-activation ``x`` as the analysis state for leaky models and hidden
activation ``h`` for Elman models. The adapter handles their different forward
signatures and return types internally.

Migrating From Separate Analysis Classes
----------------------------------------

Replace ``emFixedPointFinder``, ``emFlowFieldFinder``, and ``emLinearization``
with ``mFixedPointFinder``, ``mFlowFieldFinder``, and ``mLinearization``.
These names accept either model; the old Elman analysis exports were removed.
The implementation modules are now ``mrnntorch.analysis.fp_finder``,
``mrnntorch.analysis.flow_field_finder``, and ``mrnntorch.analysis.linear``.
The former ``fixed_points/``, ``flow_fields/``, and ``linear/`` analysis
submodules are no longer importable.

Pass PCA data as ``fit_states=...`` to the flow finder. Call
``lin.eigendecomposition(state)`` for zero-input eigenvalues; for nonzero input,
use ``lin.jacobian(input, state)`` and decompose the returned state Jacobian.
Linearization derives leaky activity internally when ``dh=True`` and does not
accept a separate ``h`` argument. Use ``delta_state_static`` for static-region
perturbations in the shared linearization and flow finder.

The interactive visualizers retain their separate leaky and Elman classes.

.. toctree::
   :maxdepth: 1

   mrnntorch.analysis.adapter
   mrnntorch.analysis.fixed_points
   mrnntorch.analysis.linear
   mrnntorch.analysis.flow_fields
   mrnntorch.analysis.flow_visualizer
