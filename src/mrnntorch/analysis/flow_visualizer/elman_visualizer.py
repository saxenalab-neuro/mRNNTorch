import pygame
import torch
from mrnntorch.analysis.flow_fields.elman_flow_field_finder import emFlowFieldFinder
from mrnntorch.mrnn.elman_mrnn import ElmanmRNN
from rnntoolkit import FlowField
from rnntoolkit import FlowFieldFinderBase
from rnntoolkit import FlowFieldVisualizerBase
from rnntoolkit.flow_visualizer import visualizer_base
from mrnntorch.analysis.flow_visualizer.region_panel import RegionPreferencesPanel

# potentially make it easier to add preferences in future
visualizer_base.PREFERENCE.setdefault(
    "cancel_other_regions", {"choices": ("off", "on"), "fmt": str}
)

pygame.init()

CANVAS_BG = (245, 245, 250)


class emFlowFieldVisualizer(FlowFieldVisualizerBase):
    """Interactive two-dimensional viewer for an RNN's flow field.

    This class asks ``FlowFieldFinder`` to project hidden states and calculate
    motion, maintains the coordinate view during pan/zoom operations, and draws
    both the vector field and its controls. Expensive flow results are cached
    until navigation or a setting marks them as dirty.
    """

    def __init__(
        self,
        rnn: ElmanmRNN,
        num_points: int = 10,
        x_offset: int = 5,
        y_offset: int = 5,
        x_center: float = 0.0,
        y_center: float = 0.0,
        fit_states: torch.Tensor | None = None,
        axes: torch.Tensor | None = None,
        flow_type: str = "nonlinear",
        region_list: list[str] | None = None,
        cancel_other_regions: bool = False,
    ) -> None:
        """Initialize an Elman flow-field visualizer.

        Args:
            rnn: Elman mRNN whose hidden-state dynamics will be visualized.
            num_points: Number of grid samples along each plotted axis.
            x_offset: Initial half-width of the x-axis data window.
            y_offset: Initial half-width of the y-axis data window.
            x_center: Initial x-axis center in reduced coordinates.
            y_center: Initial y-axis center in reduced coordinates.
            fit_states: Full hidden states used to fit the current 2D PCA plane.
            axes: Optional explicit axes with shape ``[2, total_hidden_units]``.
            flow_type: Either ``"nonlinear"`` or ``"linear"``.
            region_list: Recurrent regions initially included in the flow plane.
            cancel_other_regions: Whether excluded recurrent regions are zeroed
                rather than held at their trajectory values.
        """
        pygame.init()
        # FlowFieldVisualizerBase.__init__ builds the finder through the
        # overridden build_finder(), so these attributes must exist before
        # entering the superclass constructor.
        self.available_regions = list(rnn.hid_regions)
        self.region_list = self._normalize_region_list(region_list)
        self.cancel_other_regions = cancel_other_regions

        super().__init__(
            rnn,
            num_points,
            x_offset,
            y_offset,
            x_center,
            y_center,
            fit_states,
            axes,
            flow_type,
        )
        self.preferences["cancel_other_regions"] = (
            "on" if self.cancel_other_regions else "off"
        )
        self.preferences_panel = RegionPreferencesPanel(self.pref_btn, self)

    def _normalize_region_list(self, region_list: list[str] | None) -> list[str]:
        """Return selected regions in model order.

        Empty or invalid selections fall back to all hidden regions so the
        finder always has a non-empty state subspace to reduce.
        """
        if not region_list:
            return list(self.available_regions)
        normalized = [
            region for region in self.available_regions if region in region_list
        ]
        return normalized or list(self.available_regions)

    def _region_axes(self) -> torch.Tensor | None:
        """Return explicit axes restricted to the selected region columns.

        The base finder expects axes to match the dimensionality of the
        region-filtered states. When no explicit axes were supplied, PCA is
        used instead and this method returns ``None``.
        """
        if self.axes is None:
            return None

        region_axes = []
        for region in self.region_list:
            start, end = self.rnn.get_region_indices(region)
            region_axes.append(self.axes[:, start:end])
        return torch.cat(region_axes, dim=1)

    def _rebuild_finder(self) -> None:
        """Rebuild the finder after a preference changes the analyzed subspace.

        Changing region selection or cancellation changes PCA dimensionality and
        static-region handling, so cached flow results are discarded.
        """
        self.pages = [self.build_finder()]
        self.current_page = 1
        self._flow_cache = None
        self._mark_dirty()

    def toggle_region(self, region: str) -> None:
        """Toggle one recurrent region in the analyzed flow plane.

        At least one region is kept enabled to avoid constructing an empty PCA
        input. Region order follows the model's hidden-region order.
        """
        if region in self.region_list:
            if len(self.region_list) == 1:
                return
            self.region_list = [r for r in self.region_list if r != region]
        else:
            self.region_list = [
                r for r in self.available_regions if r in {*self.region_list, region}
            ]
        self._rebuild_finder()

    def adjust_pref(self, key: str, direction: int) -> None:
        """Apply an options-menu preference change.

        Standard rendering preferences are handled by the inherited base class.
        Regional preferences rebuild the finder so future flow computations use
        the new region configuration.
        """
        super().adjust_pref(key, direction)
        if key == "cancel_other_regions":
            self.cancel_other_regions = self.preferences["cancel_other_regions"] == "on"
            self._rebuild_finder()

    def build_finder(self) -> emFlowFieldFinder:
        """Build an Elman flow-field finder for the current region selection.

        ``fit_states`` and explicit axes are narrowed to the selected regions
        before construction so the finder's PCA basis matches the states it
        will later reduce.
        """
        fit_states = self.fit_states
        if fit_states is not None:
            fit_states = self.rnn.get_region_activity(fit_states, *self.region_list)
        axes = self._region_axes()

        finder = emFlowFieldFinder(
            rnn=self.rnn,
            num_points=self.num_points,
            x_offset=self.x_offset,
            y_offset=self.y_offset,
            x_center=self.x_center,
            y_center=self.y_center,
            fit_states=fit_states,
            axes=axes,
            follow_traj=False,
            region_list=self.region_list,
            cancel_other_regions=self.cancel_other_regions,
        )
        return finder

    def prepare_data(
        self,
        inputs: torch.Tensor,
        states: torch.Tensor,
        delta_inputs: torch.Tensor | None = None,
        delta_h_static: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Flatten inputs and states into page-aligned ``[N, D]`` arrays.

        Args:
            inputs: Input sequence or batch of input sequences.
            states: Elman hidden-state sequence aligned with ``inputs``.
            delta_inputs: Optional input perturbations for linear flow fields.
            delta_h_static: Optional perturbations for excluded recurrent regions.

        Returns:
            Flattened inputs, states, input perturbations, and static-region
            perturbations. Optional perturbation outputs remain ``None`` when
            omitted.
        """
        delta_inp_nxd = (
            FlowFieldFinderBase._nxd(delta_inputs) if delta_inputs is not None else None
        )
        delta_h_static_nxd = (
            FlowFieldFinderBase._nxd(delta_h_static)
            if delta_h_static is not None
            else None
        )
        return (
            FlowFieldFinderBase._nxd(inputs),
            FlowFieldFinderBase._nxd(states),
            delta_inp_nxd,
            delta_h_static_nxd,
        )

    def compute_flow_field(
        self,
        inp_nxd: torch.Tensor,
        states_nxd: torch.Tensor,
        stim_input: torch.Tensor | None = None,
        W: torch.Tensor | None = None,
        delta_inp_nxd: torch.Tensor | None = None,
        delta_h_static_nxd: torch.Tensor | None = None,
    ) -> FlowField:
        """Compute and cache the flow field for the current page.

        Args:
            inp_nxd: Flattened input samples.
            states_nxd: Flattened Elman hidden-state samples.
            stim_input: Optional additive stimulus used by nonlinear flow fields.
            W: Optional recurrent weight matrix override.
            delta_inp_nxd: Optional flattened input perturbations.
            delta_h_static_nxd: Optional flattened static-region perturbations.

        Returns:
            The computed flow field for ``current_element_idx``.

        The active finder is synchronized with the current viewport before each
        computation so panning, zooming, and grid-density changes affect the
        next flow calculation.
        """
        state_n = states_nxd[self.current_element_idx]
        inp_n = inp_nxd[self.current_element_idx]
        delta_inp_n = (
            delta_inp_nxd[self.current_element_idx]
            if delta_inp_nxd is not None
            else None
        )
        delta_h_static_n = (
            delta_h_static_nxd[self.current_element_idx]
            if delta_h_static_nxd is not None
            else None
        )

        finder = self.current_field()
        finder.num_points = self.preferences["grid_points"]
        finder.x_offset = self.view_span / 2.0
        finder.y_offset = self.view_span / 2.0
        # Keep the finder grid exactly aligned with the viewport. Snapping
        # this center causes gaps or apparent heatmap motion after panning.
        finder.x_center = (self.x_bounds[0] + self.x_bounds[1]) / 2.0
        finder.y_center = (self.y_bounds[0] + self.y_bounds[1]) / 2.0

        if state_n.dim() == 1:
            state_n = state_n.unsqueeze(0)
        if inp_n.dim() == 1:
            inp_n = inp_n.unsqueeze(0)

        with torch.no_grad():
            if self.flow_type == "linear":
                if delta_inp_n is None:
                    delta_inp_n = torch.zeros_like(inp_n)
                if delta_inp_n.dim() == 1:
                    delta_inp_n = delta_inp_n.unsqueeze(0)
                flow = finder.find_linear_flow(
                    state_n, inp_n, delta_inp_n, delta_h_static=delta_h_static_n
                )[0]
            else:
                flow = finder.find_nonlinear_flow(
                    state_n, inp_n, stim_input=stim_input, W=W
                )[0]

        self._flow_cache = {
            "grid": flow.grid.detach().cpu().numpy(),
            "x_vel": flow.x_vels.detach().cpu().numpy(),
            "y_vel": flow.y_vels.detach().cpu().numpy(),
            "speed": flow.speeds.detach().cpu().numpy(),
        }
        self._flow_dirty = False
        return flow

    def visualize(
        self,
        inputs: torch.Tensor,
        states: torch.Tensor,
        stim_input: torch.Tensor | None = None,
        W: torch.Tensor | None = None,
        delta_inputs: torch.Tensor | None = None,
        delta_h_static: torch.Tensor | None = None,
    ) -> None:
        """Run the interactive Elman flow-field visualization loop.

        Args:
            inputs: Input sequence or batch of sequences shown page by page.
            states: Hidden states aligned with ``inputs``.
            stim_input: Optional additive stimulus used by nonlinear flow fields.
            W: Optional recurrent weight matrix override.
            delta_inputs: Optional input perturbations for linear flow fields.
            delta_h_static: Optional perturbations for excluded recurrent regions.

        The method exits when the Pygame window is closed. Region selections
        are applied dynamically inside the loop before drawing each frame.
        """
        inp_nxd, states_nxd, delta_inp_nxd, delta_h_static_nxd = self.prepare_data(
            inputs, states, delta_inputs=delta_inputs, delta_h_static=delta_h_static
        )
        self.n_pages = inp_nxd.shape[0]
        if states_nxd.shape[0] != self.n_pages:
            raise ValueError("inputs and states must contain the same number of pages")

        while self.running:
            self.handle_events()
            draw_states = self.rnn.get_region_activity(states_nxd, *self.region_list)
            if self._flow_dirty or self._flow_cache is None:
                self.compute_flow_field(
                    inp_nxd,
                    states_nxd,
                    stim_input=stim_input,
                    W=W,
                    delta_inp_nxd=delta_inp_nxd,
                    delta_h_static_nxd=delta_h_static_nxd,
                )
            self.screen.fill(CANVAS_BG)
            self.draw_grid(inp_nxd, draw_states)
            self.draw_axes()
            self.draw_top_bar()
            self.draw_toolbar()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        return
