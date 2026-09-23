"""Analyze the trained leaky flip-flop mRNN: fixed points, flow, and stability.

Run with: python examples/analysis_leaky.py
The task stores three binary values, updated by signed input pulses.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

# Resolve the local package and checkpoint independently of the working directory.
example_dir = Path(__file__).resolve().parent
results_dir = example_dir / "results" / "leaky"
sys.path.insert(0, str(example_dir.parent / "src"))

from flip_flop_data import FlipFlopData
from model import LeakmRNN
from mrnntorch.analysis import mFixedPointFinder, mFlowFieldFinder, mLinearization

# Edit these settings to change the experiment.
n_bits = 3
n_exc = 21
n_inhib = 9
n_hidden = n_exc + n_inhib
n_trials = 64
n_time = 64
n_inits = 100
max_iters = 50_000


def save_figure(fig, analysis, filename):
    """Save a plot in its analysis folder and release the figure."""
    output_dir = results_dir / analysis
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_flow_arrows(ax, field):
    """Use equal-length arrows for direction and color for relative speed."""
    x_velocity = np.asarray(field.x_vels)
    y_velocity = np.asarray(field.y_vels)
    magnitude = np.hypot(x_velocity, y_velocity)
    moving = magnitude > 0
    # Stationary points have no direction, so omit their arrows.
    x_direction = np.divide(
        x_velocity, magnitude, out=np.zeros_like(x_velocity), where=moving
    )
    y_direction = np.divide(
        y_velocity, magnitude, out=np.zeros_like(y_velocity), where=moving
    )
    ax.set_aspect("equal", adjustable="box")
    return ax.quiver(
        field.grid[..., 0],
        field.grid[..., 1],
        np.ma.array(x_direction, mask=~moving),
        np.ma.array(y_direction, mask=~moving),
        field.speeds,
        angles="xy",
        scale_units="width",
        scale=25,
        cmap="viridis",
        clim=(0, 1),
        alpha=0.85,
    )


def load_model():
    """Load the trained checkpoint on CPU."""
    checkpoint = example_dir / "flip_flop_rnn_leaky.pth"
    model = LeakmRNN(n_bits, n_exc, n_inhib, n_bits, device="cpu").cpu()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    print(model)
    return model


def plot_constrained_weights(model):
    """Plot the effective recurrent weights used by the network dynamics."""
    rnn = model.rnn
    with torch.no_grad():
        if rnn.rec_constrained:
            weights = rnn.apply_dales_law(
                rnn.W_rec, rnn.W_rec_mask, rnn.W_rec_sign_matrix
            )
        else:
            weights = rnn.W_rec * rnn.W_rec_mask
        weights = weights.cpu().numpy()

    limit = float(np.abs(weights).max()) or 1.0
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(
        weights, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest"
    )
    ax.set(
        title="Effective recurrent weight matrix",
        xlabel="Source neuron", ylabel="Target neuron",
    )
    # Mark the boundary between excitatory and inhibitory populations.
    boundary = rnn.get_region_indices("exc")[1] - 0.5
    ax.axvline(boundary, color="black", linewidth=0.8)
    ax.axhline(boundary, color="black", linewidth=0.8)
    fig.colorbar(image, ax=ax, label="Weight")
    save_figure(fig, "weights", "constrained_recurrent_weight_matrix.png")


def collect_trajectories(model):
    """Generate flip-flop trials and record the model response."""
    data_gen = FlipFlopData(n_bits=n_bits, n_time=n_time)
    data = data_gen.generate_data(n_trials=n_trials)
    inputs = torch.from_numpy(data["inputs"])
    h0 = torch.zeros(n_trials, n_hidden)
    x0 = torch.zeros(n_trials, n_hidden)
    with torch.no_grad():
        outputs, x_traj, h_traj = model(inputs, x0, h0)

    print("inputs:", inputs.shape)
    print("outputs:", outputs.shape)
    print("hidden trajectories:", h_traj.shape)
    return data, outputs, x_traj, h_traj


def plot_trials(data, outputs):
    """Plot task targets alongside the model predictions."""
    fig = FlipFlopData.plot_trials(data, outputs.numpy(), show=False)
    save_figure(fig, "trials", "flip_flop_targets_and_predictions.png")


def plot_region_pca(model, h_traj):
    """Plot all activity trajectories in separate PCA spaces for each cell group."""
    groups = (
        ("excitatory", ("exc",)),
        ("inhibitory", ("inhib",)),
        ("all_cells", ("exc", "inhib")),
    )
    for name, regions in groups:
        activity = model.rnn.get_region_activity(h_traj, *regions).detach().cpu()
        pca = PCA(n_components=3)
        projected = pca.fit_transform(activity.reshape(-1, activity.shape[-1]))
        projected = projected.reshape(*activity.shape[:-1], 3)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        for trial in projected:
            ax.plot(*trial.T, alpha=0.5, linewidth=0.7)
        ax.set(
            title=f"{name.replace('_', ' ').capitalize()} activity trajectories (PCA)",
            xlabel="PC1", ylabel="PC2", zlabel="PC3",
        )
        save_figure(fig, "region_pca", f"{name}_activity_trajectories_pca.png")


def find_fixed_points(model, state_traj, optimize_h=False):
    """Find zero-input fixed points from sampled trajectory states."""
    fp_finder = mFixedPointFinder(
        model.rnn,
        tol_unique=1.,
        tol_q=1e-10,
        tol_dq=1e-18,
        max_iters=max_iters,
        verbose=True,
    )
    init_states = fp_finder.sample_states(state_traj, n_inits=n_inits, noise_scale=0.1)
    ext_inp = torch.zeros(n_bits)
    unique_fps, all_fps = fp_finder.find_fixed_points(
        init_states, ext_inp, optimize_h=optimize_h
    )
    print("Unique fixed points:", unique_fps.n)
    return unique_fps, all_fps


def project_states(state_traj, fp_states):
    """Project trajectories and fixed points into the same 3D PCA coordinates."""
    # Fit to trajectories so the projection works even with fewer than three FPs.
    pca = PCA(n_components=3)
    traj_proj = pca.fit_transform(state_traj.reshape(-1, state_traj.shape[-1]))
    traj_proj = traj_proj.reshape(*state_traj.shape[:-1], 3)
    fp_proj = pca.transform(fp_states) if len(fp_states) else np.empty((0, 3))
    return traj_proj, fp_proj


def plot_fixed_points(
    state_traj,
    fp_states,
    title="Fixed points (PCA space)",
    color="blue",
    stability=None,
    analysis="fixed_points",
):
    """Plot fixed points and trajectories, optionally colored by stability."""
    traj_proj, fp_proj = project_states(state_traj, fp_states)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    colors = (
        color
        if stability is None
        else ["blue" if stable else "red" for stable in stability]
    )
    ax.scatter(*fp_proj.T, c=colors, s=40, label="fixed points")
    for trial in traj_proj:
        ax.plot(*trial.T, color="black", alpha=0.4, linewidth=0.5)
    ax.set(title=title, xlabel="PC1", ylabel="PC2", zlabel="PC3")
    ax.legend()
    save_figure(fig, analysis, f"{analysis}_and_trajectories_pca.png")


def analyze_activity_fixed_points(model, x_traj, h_traj):
    """Use an activity residual while still optimizing pre-activation states."""
    unique_fps, all_fps = find_fixed_points(model, x_traj, optimize_h=True)
    fp_activity = model.rnn.activation(unique_fps.xstar)
    plot_fixed_points(
        h_traj,
        fp_activity,
        title="Activity fixed points (PCA space)",
        color="green",
        analysis="activity_fixed_points",
    )
    return unique_fps, all_fps


def analyze_nonlinear_flow(model, state_traj, fp_states):
    """Compute and plot four nonlinear flow fields with fixed-point overlays."""
    flow_finder = mFlowFieldFinder(
        model.rnn,
        num_points=15,
        x_offset=2,
        y_offset=2,
        fit_states=state_traj,
        follow_traj=False,
    )
    flow_states = state_traj.reshape(-1, state_traj.shape[-1])[:4]
    flow_inputs = torch.zeros(flow_states.shape[0], n_bits)
    flow_fields = flow_finder.find_nonlinear_flow(flow_states, flow_inputs)
    fp_proj_2d = (
        flow_finder.transform(fp_states) if len(fp_states) else np.empty((0, 2))
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        if i >= len(flow_fields):
            ax.axis("off")
            continue
        field = flow_fields[i]
        plot_flow_arrows(ax, field)
        ax.scatter(*fp_proj_2d.T, color="black", s=10)
        ax.set(title=f"Flow field t={i}", xlabel="PC1", ylabel="PC2")
    fig.suptitle("Nonlinear flow fields + fixed points (PCA space)")
    plt.tight_layout()
    save_figure(fig, "nonlinear_flow", "nonlinear_flow_and_fixed_points_pca.png")
    return flow_fields


def analyze_stability(model, state_traj, fp_states):
    """Classify discrete-time stability and retain unstable eigenvectors."""
    linearization = mLinearization(model.rnn)
    stability = []
    eigenvectors = []
    for state in fp_states:
        real, imag, eigvec = linearization.eigendecomposition(state)
        magnitude = torch.sqrt(real.square() + imag.square())
        stability.append(bool(torch.all(magnitude < 1)))
        eigenvectors.append(eigvec[:, magnitude > 1])  # Eigenvectors are columns.
    plot_fixed_points(
        state_traj,
        fp_states,
        title="Fixed-point stability: blue stable, red unstable or marginal",
        stability=stability,
        analysis="stability",
    )
    return stability, eigenvectors


def analyze_linear_flow(model, state_traj, fp_states, stability):
    """Compute and plot local linear flow around up to nine fixed points."""
    flow_finder = mFlowFieldFinder(
        model.rnn,
        num_points=15,
        x_offset=0.5,
        y_offset=0.5,
        fit_states=state_traj,
        follow_traj=True,
    )
    n_show = min(9, len(fp_states))
    linear_states = fp_states[:n_show]
    linear_inputs = torch.zeros(n_show, n_bits)
    flow_fields = flow_finder.find_linear_flow(
        linear_states, linear_inputs, torch.zeros_like(linear_inputs)
    )
    fp_proj_2d = flow_finder.transform(fp_states)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i >= n_show:
            ax.axis("off")
            continue
        field = flow_fields[i]
        plot_flow_arrows(ax, field)
        ax.scatter(
            *fp_proj_2d[i],
            color="blue" if stability[i] else "red",
            s=50,
            marker="o" if stability[i] else "x",
        )
        ax.set(title=f"Fixed point {i + 1}", xlabel="PC1", ylabel="PC2")
    fig.suptitle("Linear flow fields + fixed points (PCA space)")
    plt.tight_layout()
    save_figure(fig, "linear_flow", "linear_flow_near_fixed_points_pca.png")
    return flow_finder


def main():
    """Run each analysis in sequence."""
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    model = load_model()
    plot_constrained_weights(model)
    data, outputs, x_traj, h_traj = collect_trajectories(model)
    state_traj = x_traj
    plot_trials(data, outputs)
    plot_region_pca(model, h_traj)
    unique_fps, _ = find_fixed_points(model, state_traj)
    fp_states = unique_fps.xstar
    plot_fixed_points(state_traj, fp_states)
    analyze_activity_fixed_points(model, x_traj, h_traj)
    analyze_nonlinear_flow(model, state_traj, fp_states)
    stability, eigenvectors = analyze_stability(model, state_traj, fp_states)
    if unique_fps.n:
        flow_finder = analyze_linear_flow(model, state_traj, fp_states, stability)
    else:
        print("No fixed points found; skipping local flow and alignment plots.")


if __name__ == "__main__":
    main()
