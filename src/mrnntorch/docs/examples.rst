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

Shared Adapter And Fixed-Point Search
-------------------------------------

The following snippets continue with the Elman model above. For the leaky
model, use its ``xs`` trajectory in place of ``hs`` as the analysis state.
The analysis class names are identical for both models.

.. code-block:: python

   from mrnntorch.analysis import mRNNAdapter, mFixedPointFinder

   adapter = mRNNAdapter(rnn)
   state = adapter.batched_initial_condition(batch_size=16)
   next_state = adapter.step(inputs[:, 0], state)
   next_activity = adapter.activity(next_state)

   finder = mFixedPointFinder(rnn, max_iters=100, verbose=False)
   initial_states = finder.sample_states(hs.detach(), n_inits=32, noise_scale=0.1)
   unique_fps, all_fps = finder.find_fixed_points(initial_states, torch.zeros(2))

Region-Specific Flow Field
--------------------------

.. code-block:: python

   from mrnntorch.analysis import mFlowFieldFinder

   finder = mFlowFieldFinder(
       rnn,
       num_points=25,
       x_offset=5,
       y_offset=5,
       fit_states=hs.detach().reshape(-1, hs.shape[-1]),
       region_list=["ctx"],
       excluded_static_regions=[],
   )
   fields = finder.find_nonlinear_flow(hs.detach()[:, :1], inputs[:, :1])

Linearization Around A State
----------------------------

.. code-block:: python

   from mrnntorch.analysis import mLinearization

   lin = mLinearization(rnn, "ctx")
   jacobian, input_jacobian = lin.jacobian(inputs[0, 0], hs[0, 0])
   real_parts, imaginary_parts, eigenvectors = lin.eigendecomposition(hs[0, 0])

Interactive Flow Visualizer
---------------------------

.. code-block:: python

   from mrnntorch.analysis.flow_visualizer.elman_visualizer import emFlowFieldVisualizer

   visualizer = emFlowFieldVisualizer(
       rnn,
       num_points=25,
       fit_states=hs.detach().reshape(-1, hs.shape[-1]),
       region_list=["ctx"],
       flow_type="nonlinear",
   )
   visualizer.visualize(inputs, hs)

Runnable Analysis Scripts
-------------------------

From the repository root, run:

.. code-block:: bash

   python examples/analysis_leaky.py
   python examples/analysis_elman.py

Each script loads its matching flip-flop checkpoint, collects trajectories,
and calls separate analysis functions from ``main()``. Edit the settings near
the top of the script; there are no command-line arguments. Figures are saved
under ``examples/results/{leaky,elman}/`` in folders for trials, constrained
weights, region PCA, fixed points, stability, and linear/nonlinear flow.
The leaky script also compares activity-residual fixed points. Images have
descriptive filenames and are overwritten on reruns.

Building The Documentation
--------------------------

Install the documentation dependencies and build from the repository root:

.. code-block:: bash

   python -m pip install -r src/mrnntorch/docs/requirements.txt
   python -m pip install -e .
   python -m sphinx -b html -W --keep-going src/mrnntorch/docs /tmp/mrnntorch-docs
