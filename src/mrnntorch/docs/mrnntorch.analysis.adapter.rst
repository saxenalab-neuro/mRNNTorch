Analysis Adapter
================

``mRNNAdapter(rnn)`` presents a tensor state interface for either model.
It preserves gradients and exposes the original network as ``adapter.rnn``.
Analysis classes construct their own adapter; use it directly when writing
custom analyses.

.. list-table:: State conventions
   :header-rows: 1

   * - Model / call
     - Input state
     - Returned state
   * - Leaky, no ``h`` argument
     - Pre-activation ``x``
     - Next ``x``
   * - Leaky, explicit ``h=...``
     - Pre-activation ``x`` plus independent activity ``h``
     - Next ``h``
   * - Elman
     - Hidden activity ``h``
     - Next ``h``

``step(inp, state)`` takes ``[B, I]`` inputs and ``[B, H]`` states and returns
``[B, H]``. Do not add a time dimension: the adapter handles it, independently
of ``rnn.batch_first``. Optional ``stim_input`` has shape ``[B, H]``.
``forward(inp, state)`` and calling the adapter directly instead process
sequences in the model's batch-first or time-first layout.

``activity(state)`` applies the leaky activation, or returns an Elman state
unchanged. Do not apply it to a leaky result obtained with explicit ``h``:
that result is already activity. Elman models reject the separate ``h`` keyword.
Noise is disabled by default; ``W_rec`` supplies an effective recurrent matrix
override, following the underlying model's convention.

.. automodule:: mrnntorch.analysis.adapter
   :members:
   :show-inheritance:
