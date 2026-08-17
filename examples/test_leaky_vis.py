"""Visualize a two-region leaky mRNN with simple attractor dynamics.

The ``line`` region has one neutral mode and one contracting mode under the
leaky update.  The ``point`` region is driven to the origin.  Cross-region
connections are randomized because they are not part of this demonstration.
"""

import torch

from mrnntorch import mRNN
from mrnntorch.analysis.flow_visualizer.leaky_visualizer import mFlowFieldVisualizer


def build_network(seed: int = 7) -> mRNN:
    """Construct the two-region leaky network and install the desired weights."""
    torch.manual_seed(seed)

    rnn = mRNN(
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

    # W_rec is arranged as [destination units, source units]. With linear
    # activation, W_rec eigenvalue 1 gives a neutral mode for the leaky update.
    with torch.no_grad():
        rnn.W_rec.zero_()
        rnn.W_rec[:2, :2] = torch.diag(torch.tensor([1.0, 0.5]))
        rnn.W_rec[:2, 2:] = 0.05 * torch.randn(2, 2)
        rnn.W_rec[2:, :2] = 0.05 * torch.randn(2, 2)
        # The zero block for point -> point makes the origin an attractor.

        rnn.W_inp.zero_()
        rnn.W_inp[0, 0] = 0.05  # constant input integrates along the line

    return rnn


def gather_trials(
    rnn: mRNN, n_trials: int = 12, n_steps: int = 80
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run trials with different initial positions and a constant input."""
    initial_line_positions = torch.linspace(-1.0, 1.0, n_trials)
    x0 = torch.zeros(n_trials, 4)
    x0[:, 0] = initial_line_positions
    h0 = rnn.activation(x0)
    inputs = torch.full((n_trials, n_steps, 1), 1.0)
    xs, _ = rnn(inputs, x0, h0, noise=False)
    return inputs, xs


def main() -> None:
    rnn = build_network()
    inputs, states = gather_trials(rnn)
    inputs = inputs.detach()
    states = states.detach()

    # Restrict the reduced plane to the line-attractor region. The leaky
    # visualizer consumes x-states by default, matching mFlowFieldFinder.
    visualizer = mFlowFieldVisualizer(
        rnn,
        num_points=25,
        fit_states=states.reshape(-1, states.shape[-1]),
        region_list=["line"],
        flow_type="nonlinear",
    )
    visualizer.visualize(inputs, states)


if __name__ == "__main__":
    main()
