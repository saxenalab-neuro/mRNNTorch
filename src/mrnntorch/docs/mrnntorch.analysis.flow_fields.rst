Flow-Field Analysis
===================

``mFlowFieldFinder`` supports either model. Supply ``fit_states=...`` for PCA
or ``axes`` of shape ``[2, selected_units]``. Region selection uses
``region_list``; fitting data and axes must match the selected region features.

Pass full model states and aligned external inputs to ``find_nonlinear_flow``
or ``find_linear_flow``. Leading sample dimensions are flattened. Each method
returns one ``FlowField`` per sampled state. Default coordinates are leaky
pre-activation ``x`` or Elman hidden activity ``h``.

Leaky nonlinear flow optionally supports ``x_is_h=True``: it uses each grid
point as both initial ``x`` and ``h`` and returns activity flow. This is an
approximation, not an inverse activation. ``find_linear_flow`` operates in the
default state coordinates and does not expose a ``dh`` argument.

.. automodule:: mrnntorch.analysis.flow_field_finder
   :members:
   :show-inheritance:
