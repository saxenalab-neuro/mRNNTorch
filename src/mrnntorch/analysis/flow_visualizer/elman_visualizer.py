import pygame
import sys
import torch
from mrnntorch.analysis.flow_fields.elman_flow_field_finder import emFlowFieldFinder
from rnntoolkit import FlowField
from rnntoolkit import FlowFieldFinderBase
from rnntoolkit import FlowFieldVisualizerBase

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
        rnn,
        num_points: int = 10,
        x_offset: int = 5,
        y_offset: int = 5,
        x_center: float = 0.0,
        y_center: float = 0.0,
        fit_states: torch.Tensor | None = None,
        axes: torch.Tensor | None = None,
        flow_type: str = "nonlinear",
        region_list: list = [],
        cancel_other_regions: bool = False,
    ):
        pygame.init()
        # FlowFieldVisualizerBase.__init__ builds the finder through the
        # overridden build_finder(), so these attributes must exist before
        # entering the superclass constructor.
        self.region_list = region_list
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

    def adjust_pref(self, key, direction):
        """Update preferences and rebuild the finder for regional options."""
        super().adjust_pref(key, direction)
        if key == "cancel_other_regions":
            self.cancel_other_regions = (
                self.preferences["cancel_other_regions"] == "on"
            )
            self.pages = [self.build_finder()]
            self.current_page = 1
            self._flow_cache = None
            self._mark_dirty()

    def build_finder(self):
        """Build the RNNToolkit finder for this visualizer."""
        finder = emFlowFieldFinder(
            rnn=self.rnn,
            num_points=self.num_points,
            x_offset=self.x_offset,
            y_offset=self.y_offset,
            x_center=self.x_center,
            y_center=self.y_center,
            fit_states=self.fit_states,
            axes=self.axes,
            follow_traj=False,
            region_list=self.region_list,
            cancel_other_regions=self.cancel_other_regions,
        )
        return finder

    def prepare_data(self, inputs, states, delta_inputs=None, delta_h_static=None):
        """Flatten RNNToolkit inputs and states into page-aligned samples."""
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
        inp_nxd,
        states_nxd,
        stim_input=None,
        W=None,
        delta_inp_nxd=None,
        delta_h_static_nxd=None,
    ) -> FlowField:
        """Compute one page through the finder's public flow methods."""
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
        inputs,
        states,
        stim_input=None,
        W=None,
        delta_inputs=None,
        delta_h_static=None,
    ):
        """Prepare arbitrary data and run the shared visualization loop.

        Subclasses with a different lifecycle or rendering data contract may
        override this method entirely. Otherwise, ``prepare_data`` adapts
        their inputs into the two page-aligned arrays consumed by the common
        drawing and interaction code.
        """
        inp_nxd, states_nxd, delta_inp_nxd, delta_h_static_nxd = self.prepare_data(
            inputs, states, delta_inputs=delta_inputs, delta_h_static=delta_h_static
        )
        self.n_pages = inp_nxd.shape[0]
        if states_nxd.shape[0] != self.n_pages:
            raise ValueError("inputs and states must contain the same number of pages")

        draw_states = self.rnn.get_region_activity(states_nxd, *self.region_list)

        while self.running:
            self.handle_events()
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
