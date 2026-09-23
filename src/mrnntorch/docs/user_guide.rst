User Guide
==========

Building Networks
-----------------

Declare recurrent and input regions, add their connections, and call
``finalize_connectivity()`` to assemble weights, masks, and sign constraints.
Regions identify contiguous slices of the full recurrent state.

Choosing State Coordinates
--------------------------

``mRNN`` evolves pre-activation ``x`` and returns ``(xs, hs)`` with
``hs = activation(xs)``. ``ElmanmRNN`` evolves and returns hidden activity ``hs``.
The shared analysis classes use ``xs`` for leaky models and ``hs`` for Elman
models by default. Pass the underlying recurrent model, rather than a wrapper
that also includes an output layer.

Use ``mRNNAdapter`` for custom analyses that need the same calling convention
for both models. Its ``step`` method accepts batched inputs and states without
a time dimension. See :doc:`mrnntorch.analysis.adapter` for output conventions
when supplying independent leaky activity.

Region-Specific Analysis
------------------------

``mFlowFieldFinder`` accepts ``region_list`` and ``excluded_static_regions``.
Excluded static regions must lie outside the selected flow regions.
``mLinearization(rnn, "region1", "region2")`` accepts positional region names.
For fixed-point optimization, pass region names to
``finder.find_fixed_points(initial_states, inputs, "region1", "region2")``.
Unoptimized regions retain their supplied states.

Dimensionality Reduction
------------------------

Flow fields use a two-dimensional PCA plane fitted from ``fit_states`` or
explicit ``axes`` of shape ``[2, selected_units]``. Select the same region
features for fitting data and axes. Use detached CPU trajectories for PCA.
Fit one projection and transform both trajectories and fixed points with it
when comparing their positions.

Fixed Points
------------

Sample representative trajectories to initialize a zero-input or constant-input
search. The default residual measures the change in the model's analysis state.
Leaky ``optimize_h=True`` measures activity changes while still optimizing
pre-activation. Convert its returned ``xstar`` using the model's activation
before plotting against ``h`` trajectories. A small activity residual and a
small pre-activation residual are different numerical criteria.

Linearization
-------------

The state and input Jacobians describe one-step dynamics near an operating
point. Leaky ``dh=True`` differentiates activity while holding pre-activation
fixed; ``alpha_scaling=True`` retains the step-size factor. To classify local
stability of the full discrete state update, use the default state Jacobian:
all eigenvalue magnitudes must be below one for strict linear stability.

Flow Fields And Visualization
-----------------------------

The shared flow finder lifts a grid from the PCA plane into full state space
and evaluates nonlinear or linearized dynamics. The separate interactive
leaky and Elman visualizers still provide a Pygame UI with region selection,
panning, and zooming. They require a graphical session.

The analysis scripts in ``examples/`` instead save figures to
``examples/results/leaky/`` or ``examples/results/elman/``, with one folder per
analysis. They do not open interactive plot windows.
