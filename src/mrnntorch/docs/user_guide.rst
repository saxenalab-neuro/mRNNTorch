User Guide
==========

Building Networks
-----------------

Build networks by adding recurrent regions, optional input regions, and then
explicit connectivity. Call ``finalize_connectivity()`` after all regions and
connections are declared; this assembles full weight matrices, masks, and sign
constraints used by the forward pass.

Regions define contiguous slices in the full hidden state. Analysis tools use
these region names to select subspaces, cancel excluded regions, or combine
selected and static activity back into full model states.

Choosing A Model Type
---------------------

Use ``mRNN`` when you want leaky continuous-time-style dynamics with separate
pre-activation and activation states. Use ``ElmanmRNN`` when a single hidden
activity state is sufficient.

The leaky model returns ``(xs, hs)``. Most leaky analysis tools expect ``xs`` by
default, unless a method explicitly exposes an ``x_is_h`` option. The Elman
model returns ``hs`` and its analysis tools operate directly on hidden activity.

Region-Specific Analysis
------------------------

Most analysis classes accept ``region_list``. When supplied, the analysis is
performed in the concatenated activity of those regions, in model order. Regions
not listed can be treated as static context, or zeroed by enabling
``cancel_other_regions`` where supported.

Dimensionality Reduction
------------------------

Flow-field tools reduce selected region activity into a two-dimensional plane.
They either fit PCA from ``fit_states`` or use explicit ``axes`` with shape
``[2, selected_units]``. If you change region selection, fit states and axes
must match the new selected dimensionality.

Fixed Points
------------

Fixed-point finders search for states where one-step dynamics are stationary or
nearly stationary. Use them after collecting representative trajectories so
initial guesses cover the state-space region of interest.

Linearization
-------------

Linearization tools compute local Jacobians and eigendecompositions around a
state and input. For leaky models, be explicit about whether you are analyzing
pre-activation ``x`` dynamics or hidden activity ``h`` dynamics, because ReLU and
alpha scaling can change the interpretation of eigenvalues.

Flow Fields And Visualization
-----------------------------

Flow-field finders build a grid in a reduced two-dimensional plane, lift the
grid back into model state space, and evaluate either nonlinear or linearized
local dynamics. The visualizers wrap this process in an interactive Pygame UI
with panning, zooming, grid-size controls, rendering options, and region toggles.
