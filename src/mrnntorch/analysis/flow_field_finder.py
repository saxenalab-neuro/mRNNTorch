"""Two-dimensional flow-field estimation for leaky and Elman mRNN trajectories."""

import torch
from mrnntorch.analysis.linear import mLinearization
from rnntoolkit.flow_fields.flow_field import FlowField
from rnntoolkit.flow_fields.flow_field_finder_base import FlowFieldFinderBase
from mrnntorch.mrnn.leaky_mrnn import mRNN
from mrnntorch.mrnn.elman_mrnn import ElmanmRNN
from mrnntorch.analysis.adapter import mRNNAdapter
import warnings


class mFlowFieldFinder(FlowFieldFinderBase[mRNN]):
    """Shared flow-field estimator for leaky and Elman mRNN dynamics."""

    def __init__(
        self,
        rnn: mRNN | ElmanmRNN,
        num_points: int,
        x_offset: int,
        y_offset: int,
        x_center: int = 0,
        y_center: int = 0,
        fit_states: torch.Tensor | None = None,
        axes: torch.Tensor | None = None,
        follow_traj: bool = False,
        region_list: list = [],
        excluded_static_regions: list = [],
    ):
        """Initialize a 2D flow-field finder around a trajectory.

        Args:
            rnn (mRNN | ElmanmRNN): Network to analyze.
            fit_states (torch.Tensor): States used to fit the dimensionality
                reduction used for the flow-field plane.
            num_points (int): Number of grid points along each axis.
            x_offset (int): Horizontal half-width of the sampled grid.
            y_offset (int): Vertical half-width of the sampled grid.
            x_center (int): Fixed x-axis center when not following the trajectory.
            y_center (int): Fixed y-axis center when not following the trajectory.
            follow_traj (bool): If ``True``, center the grid on each sampled state.
            region_list (list): Recurrent regions to include in the reduced plane.
            excluded_static_regions (list): Static regions whose state input is
                zeroed. Each entry must belong to ``static_region_list``.
        """
        super().__init__(
            rnn, num_points, x_offset, y_offset, x_center, y_center, fit_states, axes
        )

        # Unload mrnn specific kwargs
        self.excluded_static_regions = excluded_static_regions
        self.follow_traj = follow_traj
        self.adapter = mRNNAdapter(self.rnn)

        self.zero_states = torch.zeros(
            size=(
                1,
                rnn.total_num_units,
            )
        )

        # Regions which are treated as grid elements
        self.region_list = (
            self.rnn.hid_regions
            if not region_list
            else list(self.rnn.ensure_order(*region_list))
        )
        # Regions treated as static inputs for grid elements
        self.static_region_list = (
            []
            if self.region_list == self.rnn.hid_regions
            else self.rnn.get_excluded_hid_regions(*self.region_list)
        )
        assert set(self.excluded_static_regions) <= set(self.static_region_list)
        self.linearization = mLinearization(rnn, *self.region_list)

    def find_nonlinear_flow(
        self,
        states: torch.Tensor,
        input: torch.Tensor,
        stim_input: torch.Tensor | None = None,
        W: torch.Tensor | None = None,
        x_is_h: bool = False,
    ) -> list:
        """Compute nonlinear 2D flow fields in a region subspace along a trajectory.

        Projects selected region activity onto a 2D PCA subspace, constructs a grid
        around the current point, and advances the system by one step to estimate
        the local flow (velocity vectors). Can zero out non-selected regions or
        keep their control values.

        Args:
            states (torch.Tensor): Full leaky x or Elman h states, with units
                in the last dimension and arbitrary leading sample dimensions.
            input (torch.Tensor): External input sequence.

        Kwargs:
            stim_input (torch.Tensor): tensor input to network without weights, acts as manipulation
            W (torch.Tensor): replace the weight matrix of mRNN with W
            x_is_h (bool): whether to assume x=h, this will give an approximation of \
                the flow field in h by assuming they are equal

        Returns:
            list: FlowField object per sampled time.
        """
        if x_is_h and not self.adapter.is_leaky:
            warnings.warn("x_is_h is True but rnn is not leaky, parameter is ignored")

        flow_field_list = []

        if stim_input is None:
            stim_input = torch.zeros_like(states, dtype=self.dtype)

        # Reshape to nxd
        states, input, stim_input = (
            self._nxd(states),
            self._nxd(input),
            self._nxd(stim_input),
        )

        assert states.shape[0] == input.shape[0]
        n_states = states.shape[0]

        # get region activity for fitting and reduction
        tmp_states = self.rnn.get_region_activity(states, *self.region_list)
        reduced_traj = self._reduce_traj(tmp_states)

        if not self.static_region_list:
            # Default to dummy tensor with shape
            static_states = None
        else:
            static_states = []
            for static_r in self.static_region_list:
                static_x = self.rnn.get_region_activity(states, static_r)
                if static_r in self.excluded_static_regions:
                    static_x = static_x * torch.zeros_like(static_x)
                static_states.append(static_x)
            static_states = torch.cat(static_states, dim=-1)

        # Now going through trajectory
        for n in range(n_states):
            # default for static states
            reduced_traj_n = reduced_traj[n]
            input_n = input[n]
            static_states_n = static_states[n] if static_states is not None else None
            stim_input_n = stim_input[n]

            # If follow trajectory is true get grid centered around current t
            # This will make a different grid for each state (n grids)
            if self.follow_traj:
                lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y = (
                    self._set_tv_bounds(reduced_traj_n)
                )
            else:
                lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y = (
                    self._set_bounds()
                )

            low_dim_grid, inverse_grid = self._inverse_grid(
                lower_bound_x,
                upper_bound_x,
                lower_bound_y,
                upper_bound_y,
            )

            # Repeat along the batch dimension to match the grid
            if static_states_n is not None:
                static_states_batch = static_states_n.repeat(low_dim_grid.shape[0], 1)
            else:
                static_states_batch = None

            full_input_batch = input_n.repeat(low_dim_grid.shape[0], 1)
            full_stim_batch = stim_input_n.repeat(low_dim_grid.shape[0], 1)

            # Combine the grid and static states to treat excluded regions as input
            if static_states_batch is not None:
                grid_flow = self.rnn.combine_states(
                    inverse_grid,
                    static_states_batch,
                    self.region_list,
                    self.static_region_list,
                )
            else:
                grid_flow = inverse_grid

            with torch.no_grad():
                next_state = self.adapter.step(
                    full_input_batch,
                    grid_flow,
                    h=grid_flow if self.adapter.is_leaky and x_is_h else None,
                    stim_input=full_stim_batch,
                    noise=False,
                    W_rec=W,
                )

            next_state = self.rnn.get_region_activity(next_state, *self.region_list)
            next_state_reduced = self._reduce_traj(next_state)

            x_vel, y_vel = self._compute_velocity(next_state_reduced, low_dim_grid)
            speed = self._compute_speed(x_vel, y_vel)

            # Reshape to match FlowField object requirements
            x_vel, y_vel, low_dim_grid, speed = self._reshape_vals(
                x_vel, y_vel, low_dim_grid, speed
            )

            flow_field = FlowField(x_vel, y_vel, low_dim_grid, speed)

            flow_field_list.append(flow_field)

        return flow_field_list

    def find_linear_flow(
        self,
        states: torch.Tensor,
        input: torch.Tensor,
        delta_input: torch.Tensor,
        delta_state_static: torch.Tensor | None = None,
    ) -> list:
        """Compute linearized 2D flow fields around sampled trajectory states.

        Uses a local linear approximation of the dynamics around each supplied
        state and input. Coordinates are leaky x or Elman h.

        Args:
            states (torch.Tensor): Network states over time.
            input (torch.Tensor): External input sequence aligned with ``states``.
            delta_input (torch.Tensor): Input perturbations for the local linear model.
            delta_state_static (torch.Tensor | None): Perturbations for recurrent regions \
                excluded from the reduced plane, in the same coordinates as states.

        Returns:
            list: FlowField objects per sampled time.
        """

        # reshape to nxd
        states, input, delta_input = self._nxd(states), self._nxd(input), self._nxd(delta_input)

        assert input.shape[0] == delta_input.shape[0]
        assert states.shape[0] == input.shape[0]
        n_states = states.shape[0]

        # Lists for x and y velocities
        flow_field_list = []

        # Activity specific to regions in region list for later computations
        region_tmp = self.rnn.get_region_activity(states, *self.region_list)
        reduced_traj = self._reduce_traj(region_tmp)

        if delta_state_static is not None:
            delta_state_static = self._nxd(delta_state_static)

        # zero out static perturbations if regions are cancelled
        if delta_state_static is not None and self.excluded_static_regions:
            mask = []
            for static_r in self.static_region_list:
                n_units = self.rnn.get_region_size(static_r)
                if static_r in self.excluded_static_regions:
                    r_mask_cur = torch.zeros(size=(1, n_units))
                else:
                    r_mask_cur = torch.ones(size=(1, n_units))
                mask.append(r_mask_cur)
            mask = torch.cat(mask, dim=-1)
            delta_state_static = delta_state_static * mask

        for n in range(n_states):
            states_n = states[n]
            reduced_traj_n = reduced_traj[n]
            input_n = input[n]
            delta_input_n = delta_input[n]
            delta_state_static_n = (
                delta_state_static[n] if delta_state_static is not None else None
            )

            # If follow trajectory is true get grid centered around current t
            # This will make a different grid for each state (n grids)
            if self.follow_traj:
                lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y = (
                    self._set_tv_bounds(reduced_traj_n)
                )
            else:
                lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y = (
                    self._set_bounds()
                )

            # Inverse the grid to pass through RNN
            low_dim_grid, inverse_grid = self._inverse_grid(
                lower_bound_x,
                upper_bound_x,
                lower_bound_y,
                upper_bound_y,
            )

            # Get a perturbation of the activity
            region_states_n = self.rnn.get_region_activity(states_n, *self.region_list)
            """
                This assumes delta state is always of x 
                This should be ok since it is a general perturbation still
            """
            delta_states = inverse_grid - region_states_n

            with torch.no_grad():
                # get next state of h or of x if dh is false
                state_next = self.linearization(
                    input_n,
                    states_n,
                    delta_input_n,
                    delta_states,
                    delta_state_static=delta_state_static_n,
                )

            # Put next h into a grid format
            state_next = self._reduce_traj(state_next)

            # Compute velocities between gathered trajectory of grid and original grid values
            x_vel, y_vel = self._compute_velocity(state_next, low_dim_grid)
            speed = self._compute_speed(x_vel, y_vel)

            x_vel, y_vel, low_dim_grid, speed = self._reshape_vals(
                x_vel, y_vel, low_dim_grid, speed
            )

            flow_field = FlowField(x_vel, y_vel, low_dim_grid, speed)

            # Reshape data back to grid
            flow_field_list.append(flow_field)

        return flow_field_list
