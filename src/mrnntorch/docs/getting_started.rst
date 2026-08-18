Getting Started
===============

Installation
------------

Install the package from the repository root in editable mode while developing:

.. code-block:: bash

   pip install -e .

Minimal Leaky mRNN
------------------

A leaky mRNN tracks both pre-activation state ``x`` and activation state ``h``.
The recurrent update is integrated with the model's ``alpha = dt / tau`` value.

.. code-block:: python

   import torch
   from mrnntorch import mRNN

   rnn = mRNN(device="cpu", rec_constrained=False, inp_constrained=False)
   rnn.add_recurrent_region("r1", num_units=2, sign="pos", init=0.0)
   rnn.add_input_region("inp", num_units=1, sign="pos")
   rnn.add_recurrent_connection("r1", "r1")
   rnn.add_input_connection("inp", "r1")
   rnn.finalize_connectivity()

   inputs = torch.zeros(4, 20, 1)
   x0, h0 = rnn.batched_initial_condition(batch_size=4)
   xs, hs = rnn(inputs, x0, h0, noise=False)

Minimal Elman mRNN
------------------

An Elman mRNN tracks a single hidden activity state and returns hidden-state
trajectories directly.

.. code-block:: python

   import torch
   from mrnntorch import ElmanmRNN

   rnn = ElmanmRNN(device="cpu", rec_constrained=False, inp_constrained=False)
   rnn.add_recurrent_region("r1", num_units=2, sign="pos", init=0.0)
   rnn.add_input_region("inp", num_units=1, sign="pos")
   rnn.add_recurrent_connection("r1", "r1")
   rnn.add_input_connection("inp", "r1")
   rnn.finalize_connectivity()

   inputs = torch.zeros(4, 20, 1)
   h0 = rnn.batched_initial_condition(batch_size=4)
   hs = rnn(inputs, h0, noise=False)

Next Steps
----------

See :doc:`user_guide` for conceptual workflows and :doc:`examples` for
copy-pastable analysis snippets.
