Linearization Analysis
======================

``mLinearization(rnn, *regions)`` supports either model. ``jacobian(input, state)``
takes one-dimensional vectors and returns the state and input Jacobians.
``eigendecomposition(state)`` uses zero input and returns real eigenvalue parts,
imaginary parts, and eigenvectors stored as columns.

For leaky models, the default is the derivative of next ``x`` with respect to
``x``. With ``dh=True``, activity is evaluated at ``activation(state)`` and
varied independently while ``x`` is held fixed, giving the derivative of next
``h`` with respect to ``h``. Set ``alpha_scaling=True`` to retain the actual
one-step derivative in this mode; the default divides the activity Jacobians
by ``rnn.alpha``. Elman models always analyze their hidden activity state.

.. automodule:: mrnntorch.analysis.linear
   :members:
   :show-inheritance:
