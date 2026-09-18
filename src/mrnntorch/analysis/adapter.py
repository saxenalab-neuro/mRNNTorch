"""A common dynamical-state interface for leaky and Elman mRNN analysis."""

import torch

from mrnntorch.mrnn.elman_mrnn import ElmanmRNN
from mrnntorch.mrnn.leaky_mrnn import mRNN


class mRNNAdapter:
    """Wrap either model with a tensor-in, tensor-out state interface.

    The analysis state is pre-activation ``x`` for a leaky mRNN and hidden
    activation ``h`` for an Elman mRNN. By default, leaky initial activity is
    derived as ``activation(x)``. Supply ``h`` explicitly to differentiate
    with respect to initial activity while holding ``x`` fixed. Use
    :meth:`activity` only for pre-activation outputs (when h is omitted).

    The original model is available as ``rnn`` for region and weight helpers.
    Calls preserve autograd and do not change model parameters or training mode.

    Example::

        adapter = mRNNAdapter(rnn)
        state = adapter.batched_initial_condition(batch_size=8)
        next_state = adapter.step(inputs, state)  # inputs: [8, I]
        flow = next_state - state
    """

    def __init__(self, rnn: mRNN | ElmanmRNN):
        if not isinstance(rnn, (mRNN, ElmanmRNN)):
            raise TypeError("rnn must be a leaky mRNN or an ElmanmRNN")
        self.rnn = rnn

    @property
    def is_leaky(self) -> bool:
        """Whether the wrapped model evolves pre-activation states."""
        return isinstance(self.rnn, mRNN)

    def batched_initial_condition(self, batch_size: int) -> torch.Tensor:
        """Return the model's initial analysis state with shape ``[B, H]``."""
        state = self.rnn.batched_initial_condition(batch_size)
        return state[0] if self.is_leaky else state

    def activity(self, state: torch.Tensor) -> torch.Tensor:
        """Convert an analysis state or state sequence to hidden activity."""
        return self.rnn.activation(state) if self.is_leaky else state

    def __call__(
        self,
        inp: torch.Tensor,
        state: torch.Tensor,
        *,
        h: torch.Tensor | None = None,
        stim_input: torch.Tensor | None = None,
        noise: bool = False,
        W_rec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Alias for :meth:`forward`."""
        return self.forward(
            inp, state, h=h, stim_input=stim_input, noise=noise, W_rec=W_rec
        )

    def forward(
        self,
        inp: torch.Tensor,
        state: torch.Tensor,
        *,
        h: torch.Tensor | None = None,
        stim_input: torch.Tensor | None = None,
        noise: bool = False,
        W_rec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a state sequence starting from ``state`` of shape ``[B, H]``.

        Input and output sequences follow the model's ``batch_first`` setting:
        ``[B, T, features]`` or ``[T, B, features]``. Optional stimulation has
        hidden feature size ``H``. Noise is disabled by default for analysis.
        ``W_rec`` is an effective recurrent-weight override, as in the model.

        ``h`` optionally supplies independent initial leaky activity ``[B, H]``.
        When supplied, the output is hidden activity rather than pre-activation.
        It is passed through without detaching, preserving its derivatives.
        Elman models already receive activity through ``state`` and reject ``h``.
        """
        if h is not None:
            if not self.is_leaky:
                raise ValueError("For Elman models, pass activity as state, not h")
            if h.shape != state.shape:
                raise ValueError("h must have the same [B, H] shape as state")
        if self.is_leaky:
            xs, hs = self.rnn(
                inp, state, h0=h, stim_input=stim_input, noise=noise, W_rec=W_rec
            )
            return hs if h is not None else xs
        return self.rnn(
            inp, state, stim_input=stim_input, noise=noise, W_rec=W_rec
        )

    def step(
        self,
        inp: torch.Tensor,
        state: torch.Tensor,
        *,
        h: torch.Tensor | None = None,
        stim_input: torch.Tensor | None = None,
        noise: bool = False,
        W_rec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance ``[B, H]`` states using ``[B, I]`` inputs by one time step.

        Optional stimulation has shape ``[B, H]``. The returned state has
        shape ``[B, H]`` regardless of the model's sequence layout.
        ``h`` has the same meaning as in :meth:`forward`. To compute the
        leaky activity Jacobian with ``x`` fixed::

            J_h = torch.autograd.functional.jacobian(
                lambda h: adapter.step(inp, x, h=h), h
            )

        This is the discrete-step derivative, including the model's alpha.
        """
        if inp.ndim != 2 or state.ndim != 2:
            raise ValueError("inp and state must have shapes [B, I] and [B, H]")
        if stim_input is not None and stim_input.ndim != 2:
            raise ValueError("stim_input must have shape [B, H]")
        time_dim = 1 if self.rnn.batch_first else 0
        stimulation = (
            None if stim_input is None else stim_input.unsqueeze(time_dim)
        )
        return self.forward(
            inp.unsqueeze(time_dim),
            state,
            h=h,
            stim_input=stimulation,
            noise=noise,
            W_rec=W_rec,
        ).squeeze(time_dim)
