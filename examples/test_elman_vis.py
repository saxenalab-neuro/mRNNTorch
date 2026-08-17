"""Visualize a two-region Elman mRNN with simple attractor dynamics.

The ``line`` region is an identity map, so its two coordinates form a line
attractor.  The ``point`` region is driven to the origin.  Cross-region
connections are randomized because they are not part of this demonstration.
"""

import torch

from mrnntorch import ElmanmRNN
from mrnntorch.analysis.flow_visualizer.elman_visualizer import emFlowFieldVisualizer


def build_network(seed: int = 7) -> ElmanmRNN:
    """Construct the two-region network and install the desired weights."""
    torch.manual_seed(seed)

    rnn = ElmanmRNN(
        activation="linear",
        device="cpu",
        noise_level_act=0.0,
        noise_level_inp=0.0,
        rec_constrained=False,
        inp_constrained=False,
    )
    rnn.add_recurrent_region("line", num_units=2, sign="pos", init=0.0)
    rnn.add_recurrent_region("point", num_units=2, sign="pos", init=0.0)
    rnn.add_input_region("static", num_units=1, sign="pos")

    for source in ("line", "point"):
        for destination in ("line", "point"):
            rnn.add_recurrent_connection(source, destination)
    rnn.add_input_connection("static", "line")
    rnn.add_input_connection("static", "point")
    rnn.finalize_connectivity()

    # W_rec is arranged as [destination units, source units].
    with torch.no_grad():
        rnn.W_rec.zero_()
        # One neutral direction and one contracting direction: a line
        # attractor embedded in the two-neuron region.
        rnn.W_rec[:2, :2] = torch.diag(torch.tensor([1.0, 0.5]))
        rnn.W_rec[:2, 2:] = 0.05 * torch.randn(2, 2)
        rnn.W_rec[2:, :2] = 0.05 * torch.randn(2, 2)
        # The zero block for point -> point makes the origin a discrete attractor.

        rnn.W_inp.zero_()
        rnn.W_inp[0, 0] = 0.05  # constant input integrates along the line

    return rnn


def gather_trials(
    rnn: ElmanmRNN, n_trials: int = 12, n_steps: int = 80
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run trials with different initial positions and a constant input."""
    initial_line_positions = torch.linspace(-1.0, 1.0, n_trials)
    h0 = torch.zeros(n_trials, 4)
    h0[:, 0] = initial_line_positions
    inputs = torch.full((n_trials, n_steps, 1), 1.0)
    states = rnn(inputs, h0, noise=False)
    return inputs, states


def main() -> None:
    rnn = build_network()
    inputs, states = gather_trials(rnn)
    inputs = inputs.detach()
    states = states.detach()
    fit_states = rnn.get_region_activity(states, "line")

    # Restrict the reduced plane to the line-attractor region.  No delta
    # inputs are supplied; the visualizer therefore shows the ordinary field.
    visualizer = emFlowFieldVisualizer(
        rnn,
        num_points=25,
        fit_states=fit_states.reshape(-1, fit_states.shape[-1]),
        region_list=["line"],
        flow_type="nonlinear",
    )
    visualizer.visualize(inputs, states)


if __name__ == "__main__":
    main()
