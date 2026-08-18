Examples
========

The examples below show common workflows. They are intentionally short so they
can be copied into scripts or notebooks and adapted to a specific model.

Build And Run A Leaky mRNN
--------------------------

.. code-block:: python

   import torch
   from mrnntorch import mRNN

   rnn = mRNN(device="cpu", rec_constrained=False, inp_constrained=False)
   rnn.add_recurrent_region("ctx", num_units=8, sign="pos", init=0.0)
   rnn.add_input_region("stim", num_units=2, sign="pos")
   rnn.add_recurrent_connection("ctx", "ctx")
   rnn.add_input_connection("stim", "ctx")
   rnn.finalize_connectivity()

   inputs = torch.randn(16, 50, 2)
   x0, h0 = rnn.batched_initial_condition(batch_size=16)
   xs, hs = rnn(inputs, x0, h0, noise=False)

Build And Run An Elman mRNN
---------------------------

.. code-block:: python

   import torch
   from mrnntorch import ElmanmRNN

   rnn = ElmanmRNN(device="cpu", rec_constrained=False, inp_constrained=False)
   rnn.add_recurrent_region("ctx", num_units=8, sign="pos", init=0.0)
   rnn.add_input_region("stim", num_units=2, sign="pos")
   rnn.add_recurrent_connection("ctx", "ctx")
   rnn.add_input_connection("stim", "ctx")
   rnn.finalize_connectivity()

   inputs = torch.randn(16, 50, 2)
   h0 = rnn.batched_initial_condition(batch_size=16)
   hs = rnn(inputs, h0, noise=False)

Region-Specific Flow Field
--------------------------

.. code-block:: python

   from mrnntorch.analysis import emFlowFieldFinder

   finder = emFlowFieldFinder(
       rnn,
       num_points=25,
       x_offset=5,
       y_offset=5,
       fit_states=hs.reshape(-1, hs.shape[-1]),
       region_list=["ctx"],
       cancel_other_regions=False,
   )
   fields = finder.find_nonlinear_flow(hs[:, :1], inputs[:, :1])

Linearization Around A State
----------------------------

.. code-block:: python

   from mrnntorch.analysis import emLinearization

   lin = emLinearization(rnn, "ctx")
   jacobian, input_jacobian = lin.jacobian(inputs[0, 0], hs[0, 0])
   real_parts, imaginary_parts, eigenvectors = lin.eigendecomposition(
       inputs[0, 0], hs[0, 0]
   )

Interactive Flow Visualizer
---------------------------

.. code-block:: python

   from mrnntorch.analysis.flow_visualizer.elman_visualizer import emFlowFieldVisualizer

   visualizer = emFlowFieldVisualizer(
       rnn,
       num_points=25,
       fit_states=hs.reshape(-1, hs.shape[-1]),
       region_list=["ctx"],
       flow_type="nonlinear",
   )
   visualizer.visualize(inputs, hs)

See the ``examples/`` directory in the repository for runnable scripts.
