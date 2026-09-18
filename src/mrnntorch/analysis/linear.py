"""Local Jacobian and eigendecomposition tools for leaky and Elman mRNNs."""

import torch
from mrnntorch.mrnn.leaky_mrnn import mRNN
from mrnntorch.mrnn.elman_mrnn import ElmanmRNN
from mrnntorch.analysis.adapter import mRNNAdapter
from typing import Tuple
import warnings


class mLinearization:
    """Local linear analysis utilities for leaky and Elman mRNN models."""

    def __init__(
        self,
        rnn: mRNN | ElmanmRNN,
        *args,
    ):
        """Initialize the linearization helper for a model and region subset.

        Args:
            rnn (mRNN): Network to analyze.
            *args (str): Optional recurrent region names to include in the
                linearized subspace. If omitted, all recurrent regions are used.
        """
        self.rnn = rnn
        self.adapter = mRNNAdapter(rnn)
        # Regions which are treated as grid elements
        self.zero_states = torch.zeros(
            size=(
                1,
                rnn.total_num_units,
            )
        )
        self.region_list = (
            self.rnn.hid_regions
            if not args
            else [region for region in rnn.ensure_order(*args)]
        )
        # Regions treated as static inputs for grid elements
        self.static_region_list = (
            []
            if self.region_list == self.rnn.hid_regions
            else self.rnn.get_excluded_hid_regions(*self.region_list)
        )

    def __call__(
        self,
        input: torch.Tensor,
        state: torch.Tensor,
        delta_input: torch.Tensor,
        delta_state: torch.Tensor,
        *,
        delta_state_static: torch.Tensor | None = None,
        dh: bool = False,
        alpha_scaling: bool = False,
    ) -> torch.Tensor:
        """Alias for :meth:`forward`."""
        return self.forward(
            input,
            state,
            delta_input,
            delta_state,
            delta_state_static=delta_state_static,
            dh=dh,
            alpha_scaling=alpha_scaling,
        )

    def forward(
        self,
        input: torch.Tensor,
        state: torch.Tensor,
        delta_input: torch.Tensor,
        delta_state: torch.Tensor,
        *,
        delta_state_static: torch.Tensor | None = None,
        dh: bool = False,
        alpha_scaling: bool = False,
    ) -> torch.Tensor:
        """Evaluate the linear approximation in the requested state coordinates.

        Args:
            input (torch.Tensor): External input at the operating point.
            state (torch.Tensor): Leaky pre-activation or Elman hidden state about which to linearize.
            delta_input (torch.Tensor): Input perturbation.
            delta_state (torch.Tensor): Perturbation for the state of the included \
                region subset. Should be x perturbations or h perturbations if dh=True
            delta_state_static (torch.Tensor | None): Perturbation applied to excluded \
                regions when only a subset of regions is linearized.
            dh (bool): If ``True``, linearize the hidden activation update instead \
                of the pre-activation update.
            alpha_scaling (bool): Passed to :meth:`jacobian`.

        Returns:
            torch.Tensor: Linearized next state in the requested coordinates.
        """

        # Assert correct shapes
        assert input.dim() == 1
        assert state.dim() == 1

        # Flatten delta since it can be batched
        if delta_state.dim() > 1:
            delta_state = delta_state.flatten(start_dim=0, end_dim=-2)

        # Get jacobians for included regions
        _jacobian, _jacobian_inp = self.jacobian(
            input, state, dh=dh, alpha_scaling=alpha_scaling
        )
        if len(self.static_region_list) >= 1:
            # Get jacobians for excluded regions if available
            _jacobian_exc, _ = self.jacobian(
                input,
                state,
                excluded_regions=True,
                dh=dh,
                alpha_scaling=alpha_scaling,
            )
        else:
            _jacobian_exc = None

        activity = self._initial_activity(state, dh)
        out = self.adapter.step(
            input.unsqueeze(0), state.unsqueeze(0),
            h=None if activity is None else activity.unsqueeze(0),
        )

        out = self.rnn.get_region_activity(out, *self.region_list)

        if _jacobian_exc is None or delta_state_static is None:
            pert = (
                out.squeeze(0)
                + delta_state @ _jacobian.T
                + (_jacobian_inp @ delta_input)
            )
        else:
            pert = (
                out.squeeze(0)
                + delta_state @ _jacobian.T
                + (delta_state_static @ _jacobian_exc.T)
                + (_jacobian_inp @ delta_input)
            )

        return pert

    def jacobian(
        self,
        input: torch.Tensor,
        state: torch.Tensor,
        *,
        excluded_regions: bool = False,
        dh: bool = False,
        alpha_scaling: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return Jacobians of the update with respect to state and input.

        Args:
            input (torch.Tensor): Input vector at which to linearize.
            state (torch.Tensor): Leaky pre-activation or Elman hidden state at which to linearize.
            excluded_regions (bool): If ``True``, return the projection from
                excluded recurrent regions into the included region subset.
            dh (bool): If ``True``, differentiate the hidden activation output
                rather than the pre-activation state output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Jacobian with respect to state
            followed by Jacobian with respect to input. For leaky dh=True,
            h is evaluated at activation(state) with x held fixed.
            alpha_scaling=True returns the actual discrete-step
            derivatives; False divides these activity derivatives by alpha.
        """

        assert isinstance(excluded_regions, bool)
        assert state.dim() == 1
        assert input.dim() == 1
        activity = self._initial_activity(state, dh)

        def update(inp, variable):
            # Differentiate h independently, holding x fixed, in activity mode.
            next_state = self.adapter.step(
                inp.unsqueeze(0),
                (state if activity is not None else variable).unsqueeze(0),
                h=variable.unsqueeze(0) if activity is not None else None,
            )
            return next_state.squeeze(0)

        _jacobian_input, _jacobian = torch.autograd.functional.jacobian(
            update, (input, state if activity is None else activity)
        )

        # Preserve the existing optional normalization for leaky activity.
        if activity is not None and not alpha_scaling:
            _jacobian = _jacobian / self.rnn.alpha
            _jacobian_input = _jacobian_input / self.rnn.alpha

        if excluded_regions and len(self.static_region_list) >= 1:
            excluded_to_included = []
            for r_i in self.region_list:
                excluded_to_region = []
                for r_e in self.static_region_list:
                    to_start, to_end = self.rnn.get_region_indices(r_i)
                    from_start, from_end = self.rnn.get_region_indices(r_e)
                    projection = _jacobian[to_start:to_end, from_start:from_end]
                    excluded_to_region.append(projection)
                excluded_to_region = torch.cat(excluded_to_region, dim=-1)
                excluded_to_included.append(excluded_to_region)
            _jacobian = torch.cat(excluded_to_included, dim=0)
        else:
            _jacobian = self.rnn.get_weight_subset(*self.region_list, W=_jacobian)

        # Get subsets for input jacobians
        input_to_rec = []
        for r_i in self.region_list:
            input_to_region = []
            for r_e in self.rnn.inp_dict:
                to_start, to_end = self.rnn.get_region_indices(r_i)
                from_start, from_end = self.rnn.get_region_indices(r_e)
                projection = _jacobian_input[to_start:to_end, from_start:from_end]
                input_to_region.append(projection)
            input_to_region = torch.cat(input_to_region, dim=-1)
            input_to_rec.append(input_to_region)
        _jacobian_input = torch.cat(input_to_rec, dim=0)

        return _jacobian, _jacobian_input

    def eigendecomposition(
        self,
        state: torch.Tensor,
        *,
        dh: bool = False,
        alpha_scaling: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the eigendecomposition of the local Jacobian.

        Args:
            state (torch.Tensor): Leaky pre-activation or Elman hidden state at which to linearize.
            dh (bool): If ``True``, eigendecompose the hidden-state Jacobian.

        Returns:
            torch.Tensor: Real parts of eigenvalues.
            torch.Tensor: Imag parts of eigenvalues.
            torch.Tensor: Eigenvectors stacked column-wise.
        """
        input = state.new_zeros(self.rnn.total_num_inputs)
        _jacobian, _ = self.jacobian(input, state, dh=dh, alpha_scaling=alpha_scaling)
        eigenvalues, eigenvectors = torch.linalg.eig(_jacobian)

        return eigenvalues.real, eigenvalues.imag, eigenvectors

    def _initial_activity(self, state, dh):
        """Evaluate leaky activity at the supplied state for activity derivatives."""
        if not self.adapter.is_leaky:
            if dh:
                warnings.warn("dh is True but network is not leaky, option is ignored", stacklevel=3)
            return None
        return self.rnn.activation(state) if dh else None