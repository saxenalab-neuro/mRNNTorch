"""Analyze the trained elman flip-flop mRNN: fixed points, flow, and stability.

Run with: python examples/analysis_elman.py
The task stores three binary values, updated by signed input pulses.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import torch
from sklearn.decomposition import PCA

# Resolve the local package and checkpoint independently of the working directory.
example_dir = Path(__file__).resolve().parent
results_dir = example_dir / "results" / "elman"
sys.path.insert(0, str(example_dir.parent / "src"))

from flip_flop_data import FlipFlopData
from model import EmRNN
from mrnntorch.analysis import mFixedPointFinder, mFlowFieldFinder, mLinearization

# Edit these settings to change the experiment.
n_bits = 3
n_exc = 21
n_inhib = 9
n_hidden = n_exc + n_inhib
n_trials = 64
n_time = 64
n_inits = 128
max_iters = 35000


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
    checkpoint = example_dir / "flip_flop_rnn_elman.pth"
    model = EmRNN(n_bits, n_exc, n_inhib, n_bits, device="cpu").cpu()
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
        xlabel="Source neuron",
        ylabel="Target neuron",
    )
    # Mark the boundary between excitatory and inhibitory populations.
    boundary = rnn.get_region_indices("exc")[1] - 0.5
    ax.axvline(boundary, color="black", linewidth=0.8)
    ax.axhline(boundary, color="black", linewidth=0.8)
    fig.colorbar(image, ax=ax, label="Weight")
    save_figure(fig, "weights", "constrained_recurrent_weight_matrix.png")


def collect_trajectories(model, trial_count=None, time_steps=None):
    """Generate flip-flop trials and record the model response."""
    trial_count = n_trials if trial_count is None else trial_count
    time_steps = n_time if time_steps is None else time_steps
    data_gen = FlipFlopData(n_bits=n_bits, n_time=time_steps)
    data = data_gen.generate_data(n_trials=trial_count)
    inputs = torch.from_numpy(data["inputs"])
    h0 = torch.zeros(trial_count, n_hidden)
    with torch.no_grad():
        outputs, h_traj = model(inputs, h0)

    print("inputs:", inputs.shape)
    print("outputs:", outputs.shape)
    print("hidden trajectories:", h_traj.shape)
    return data, outputs, h_traj


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
            ax.plot(*trial.T, color="black", alpha=0.4, linewidth=0.25)
        ax.set(
            title=(
                ""
                if len(regions) == 1
                else f"{name.replace('_', ' ').capitalize()} activity trajectories (PCA)"
            ),
            xlabel="PC1",
            ylabel="PC2",
            zlabel="PC3",
        )
        save_figure(fig, "region_pca", f"{name}_activity_trajectories_pca.png")


def find_fixed_points(
    model,
    state_traj,
    region=None,
    static_state_t=None,
    external_input=None,
    search_n_inits=None,
    search_max_iters=None,
    lr_init=1e-4,
    initial_state_traj=None,
    init_noise=0.5,
):
    """Find fixed points at a constant external input (zero by default) from sampled trajectory states.

    With a region selected, other regions are zero unless static_state_t is
    supplied. In that case, pass one trial [time, units]; the other regions
    stay fixed at that trial's state at static_state_t.
    initial_state_traj optionally supplies a separate pool of initial guesses.
    init_noise sets the initial-state noise scale (default: 0.5).
    lr_init sets the optimizer learning rate (default: 1e-4).
    """
    initial_state_traj = (
        state_traj if initial_state_traj is None else initial_state_traj
    )
    search_n_inits = n_inits if search_n_inits is None else search_n_inits
    fp_finder = mFixedPointFinder(
        model.rnn,
        lr_init=lr_init,
        tol_unique=0.01,
        tol_q=1e-10,
        tol_dq=1e-18,
        max_iters=max_iters if search_max_iters is None else search_max_iters,
        verbose=True,
    )
    regions = () if region is None else (region,)
    if region is None:
        init_states = fp_finder.sample_states(
            initial_state_traj, n_inits=search_n_inits, noise_scale=init_noise
        )
    else:
        excluded = (
            model.rnn.get_excluded_hid_regions(region) if static_state_t is None else []
        )
        init_states = fp_finder.region_initial_states(
            initial_state_traj,
            search_n_inits,
            region,
            static_state_t=static_state_t,
            static_states=state_traj,
            noise_scale=init_noise,
            excluded_static_regions=excluded,
        )
    # Optimize only the selected region; other regions stay at their initial values.
    ext_inp = state_traj.new_zeros(n_bits) if external_input is None else external_input
    unique_fps, all_fps = fp_finder.find_fixed_points(init_states, ext_inp, *regions)
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
    filename=None,
    trajectory_end_t=None,
):
    """Plot fixed points and trajectories, optionally colored by stability."""
    traj_proj, fp_proj = project_states(state_traj, fp_states)
    # Fit PCA on the full reference trial, then truncate only the visible path.
    if trajectory_end_t is not None:
        traj_proj = traj_proj[:, : trajectory_end_t + 1]
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    colors = (
        color
        if stability is None
        else ["blue" if stable else "red" for stable in stability]
    )
    ax.scatter(*fp_proj.T, c=colors, s=40, label="fixed points")
    for trial in traj_proj:
        if len(traj_proj) == 1:
            ax.plot(
                *trial.T,
                color="black",
                linewidth=3,
                solid_capstyle="round",
                path_effects=[
                    path_effects.SimpleLineShadow(
                        offset=(2, -2),
                        shadow_color="black",
                        alpha=0.3,
                        linewidth=6,
                    ),
                    path_effects.Stroke(linewidth=5.5, foreground="white"),
                    path_effects.Normal(),
                ],
            )
            for point, marker, label in (
                (trial[0], "^", "start"),
                (trial[-1], "X", "end"),
            ):
                ax.plot(
                    [point[0]],
                    [point[1]],
                    [point[2]],
                    linestyle="none",
                    marker=marker,
                    markersize=13,
                    markerfacecolor="black",
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                    label=label,
                )
        else:
            ax.plot(*trial.T, color="black", alpha=0.3, linewidth=0.25)
    ax.set(title=title, xlabel="PC1", ylabel="PC2", zlabel="PC3")
    ax.legend()
    save_figure(fig, analysis, filename or f"{analysis}_and_trajectories_pca.png")


def analyze_nonlinear_flow(model, state_traj, fp_states):
    """Compute one zero-input nonlinear flow field with fixed-point overlays."""
    flow_finder = mFlowFieldFinder(
        model.rnn,
        num_points=15,
        x_offset=3,
        y_offset=3,
        fit_states=state_traj,
        follow_traj=False,
    )
    flow_states = state_traj.reshape(-1, state_traj.shape[-1])[:1]
    flow_inputs = torch.zeros(flow_states.shape[0], n_bits)
    flow_fields = flow_finder.find_nonlinear_flow(flow_states, flow_inputs)
    fp_proj_2d = (
        flow_finder.transform(fp_states) if len(fp_states) else np.empty((0, 2))
    )
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_flow_arrows(ax, flow_fields[0])
    ax.scatter(
        *fp_proj_2d.T,
        marker="*",
        s=350,
        facecolors="black",
        edgecolors="white",
        linewidths=1.2,
        zorder=5,
    )
    ax.set(
        title="Zero-input nonlinear flow + fixed points (PCA space)",
        xlabel="PC1",
        ylabel="PC2",
    )
    plt.tight_layout()
    save_figure(fig, "nonlinear_flow", "nonlinear_flow_and_fixed_points_pca.png")
    return flow_fields


def analyze_stability(
    model,
    state_traj,
    fp_states,
    region=None,
    label=None,
    external_input=None,
    trajectory_end_t=None,
):
    """Classify discrete-time stability and retain unstable eigenvectors."""
    regions = () if region is None else (region,)
    linearization = mLinearization(model.rnn, *regions)
    stability = []
    eigenvectors = []
    for state in fp_states:
        inp = state.new_zeros(n_bits) if external_input is None else external_input
        jacobian, _ = linearization.jacobian(inp, state)
        eigenvalues, eigvec = torch.linalg.eig(jacobian)
        magnitude = eigenvalues.abs()
        stability.append(bool(torch.all(magnitude < 1)))
        eigenvectors.append(eigvec[:, magnitude > 1])  # Eigenvectors are columns.
    plot_fixed_points(
        model.rnn.get_region_activity(state_traj, *regions),
        model.rnn.get_region_activity(fp_states, *regions),
        title=(
            ""
            if region
            else "Fixed-point stability: blue stable, red unstable or marginal"
        ),
        stability=stability,
        analysis="stability",
        trajectory_end_t=trajectory_end_t,
        filename=(
            f"{label}_stability_pca.png"
            if label
            else f"{region}_isolated_stability_pca.png" if region else None
        ),
    )
    return stability, eigenvectors


def analyze_linear_flow(
    model,
    state_traj,
    fp_states,
    stability,
    region=None,
    label=None,
    excluded_static_regions=None,
    external_input=None,
):
    """Compute and plot local linear flow around up to nine fixed points."""
    regions = [] if region is None else [region]
    excluded = (
        (model.rnn.get_excluded_hid_regions(region) if region else [])
        if excluded_static_regions is None
        else excluded_static_regions
    )
    filename = (
        f"{region}_isolated_linear_flow_near_fixed_points_pca.png"
        if region
        else "linear_flow_near_fixed_points_pca.png"
    )
    if label:
        filename = f"{label}_linear_flow_near_fixed_points_pca.png"
    if len(fp_states) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No fixed points found", ha="center", va="center")
        ax.axis("off")
        save_figure(fig, "linear_flow", filename)
        return None
    flow_finder = mFlowFieldFinder(
        model.rnn,
        num_points=15,
        x_offset=0.5,
        y_offset=0.5,
        fit_states=model.rnn.get_region_activity(state_traj, *regions),
        region_list=regions,
        excluded_static_regions=excluded,
        follow_traj=True,
    )
    n_show = min(9, len(fp_states))
    linear_states = fp_states[:n_show]
    inp = fp_states.new_zeros(n_bits) if external_input is None else external_input
    linear_inputs = inp.unsqueeze(0).expand(n_show, -1)
    flow_fields = flow_finder.find_linear_flow(
        linear_states, linear_inputs, torch.zeros_like(linear_inputs)
    )
    fp_proj_2d = flow_finder.transform(
        model.rnn.get_region_activity(fp_states, *regions)
    )
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
        ax.set(
            title="" if region else f"Fixed point {i + 1}", xlabel="PC1", ylabel="PC2"
        )
    if region is None:
        fig.suptitle("Linear flow fields + fixed points (PCA space)")
    plt.tight_layout()
    save_figure(fig, "linear_flow", filename)
    return flow_finder


def analyze_isolated_region(model, state_traj, region, init_noise=0.5):
    """Find and analyze one region with all other regions clamped to zero.

    These examples use tanh, so zero pre-activation also gives zero activity
    for the excluded region in the leaky model. The selected region retains
    its own recurrent weights and tonic input. Trajectories in the PCA plots
    are selected-region traces from the full model, provided for comparison.
    Stability uses only the selected-region Jacobian, with the other region
    fixed, rather than the full coupled network's Jacobian.
    """
    print(f"Analyzing isolated region {region}...")
    unique_fps, all_fps = find_fixed_points(
        model, state_traj, region=region, init_noise=init_noise
    )
    fp_states = unique_fps.xstar
    plot_fixed_points(
        model.rnn.get_region_activity(state_traj, region),
        model.rnn.get_region_activity(fp_states, region),
        title="",
        filename=f"{region}_isolated_fixed_points_and_trajectories_pca.png",
    )
    stability, _ = analyze_stability(model, state_traj, fp_states, region=region)
    analyze_linear_flow(model, state_traj, fp_states, stability, region=region)
    return unique_fps, all_fps, stability


def analyze_region_with_static_input(
    model,
    state_traj,
    trial_index=0,
    *,
    trial_inputs=None,
    initial_state_traj=None,
    include_external_input=False,
    time_steps=7,
    search_n_inits=16,
    search_max_iters=5000,
    region="exc",
    lr_init=1e-1,
    init_noise=0.5,
):
    """Analyze one region with the other held at each state of a short trial.

    initial_state_traj supplies the many-trial initialization pool; state_traj
    supplies the short trial for static inputs and plotting.
    Only the selected region is optimized. The other region is held at t via
    static_state_t, including during stability and flow analysis. External
    input is zero unless include_external_input=True. For leaky models, the
    static pre-activation fixes the other region's tanh activity as well.

    Return one (unique_fps, all_fps, stability) tuple per time step. PCA uses
    the selected region's trial coordinates consistently for every plot.
    """
    if region not in ("exc", "inhib"):
        raise ValueError("region must be exc or inhib")
    static_region = "inhib" if region == "exc" else "exc"
    if include_external_input and trial_inputs is None:
        raise ValueError("trial_inputs is required when include_external_input=True")
    trial_states = state_traj[trial_index, :time_steps]
    plot_states = trial_states.unsqueeze(0)
    if include_external_input and trial_inputs.shape[:2] != state_traj.shape[:2]:
        raise ValueError(
            "trial_inputs must match the state trajectories' trials and times"
        )
    condition = (
        "with_external_input" if include_external_input else "zero_external_input"
    )
    results = []
    for t in range(len(trial_states)):
        label = f"{region}_static_{static_region}_trial_{trial_index:03d}_t_{t:03d}_{condition}"
        external_input = (
            trial_inputs[trial_index, t].to(trial_states)
            if include_external_input
            else trial_states.new_zeros(n_bits)
        )
        print(f"Analyzing {label}")
        unique_fps, all_fps = find_fixed_points(
            model,
            trial_states,
            region=region,
            static_state_t=t,
            external_input=external_input,
            search_n_inits=search_n_inits,
            search_max_iters=search_max_iters,
            lr_init=lr_init,
            init_noise=init_noise,
            initial_state_traj=initial_state_traj,
        )
        fp_states = unique_fps.xstar
        plot_fixed_points(
            model.rnn.get_region_activity(plot_states, region),
            model.rnn.get_region_activity(fp_states, region),
            title="",
            trajectory_end_t=t,
            filename=f"{label}_fixed_points_and_trajectories_pca.png",
        )
        stability, _ = analyze_stability(
            model,
            plot_states,
            fp_states,
            region=region,
            label=label,
            external_input=external_input,
            trajectory_end_t=t,
        )
        analyze_linear_flow(
            model,
            plot_states,
            fp_states,
            stability,
            region=region,
            label=label,
            excluded_static_regions=[],
            external_input=external_input,
        )
        results.append((unique_fps, all_fps, stability))
    return results


def analyze_excitatory_with_static_inhibition(
    model,
    state_traj,
    trial_index=0,
    *,
    trial_inputs=None,
    initial_state_traj=None,
    include_external_input=False,
    time_steps=7,
    search_n_inits=32,
    search_max_iters=10_000,
    lr_init=1e-1,
    init_noise=0.5,
):
    """Analyze exc fixed points with the other region held at each trial state."""
    return analyze_region_with_static_input(
        model,
        state_traj,
        trial_index,
        region="exc",
        trial_inputs=trial_inputs,
        initial_state_traj=initial_state_traj,
        include_external_input=include_external_input,
        time_steps=time_steps,
        search_n_inits=search_n_inits,
        search_max_iters=search_max_iters,
        lr_init=lr_init,
        init_noise=init_noise,
    )


def analyze_inhibitory_with_static_excitation(
    model,
    state_traj,
    trial_index=0,
    *,
    trial_inputs=None,
    initial_state_traj=None,
    include_external_input=False,
    time_steps=7,
    search_n_inits=100,
    search_max_iters=10_000,
    lr_init=1e-1,
    init_noise=1.0,
):
    """Analyze inhib fixed points with the other region held at each trial state."""
    return analyze_region_with_static_input(
        model,
        state_traj,
        trial_index,
        region="inhib",
        trial_inputs=trial_inputs,
        initial_state_traj=initial_state_traj,
        include_external_input=include_external_input,
        time_steps=time_steps,
        search_n_inits=search_n_inits,
        search_max_iters=search_max_iters,
        lr_init=lr_init,
        init_noise=init_noise,
    )


def main():
    """Run each analysis in sequence."""
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    model = load_model()

    plot_constrained_weights(model)
    data, outputs, state_traj = collect_trajectories(model)

    # Use the same short trial for both regions and external-input conditions.
    short_trial = collect_trajectories(model, trial_count=1, time_steps=7)
    short_inputs = torch.from_numpy(short_trial[0]["inputs"])
    short_states = short_trial[2]
    for include_external_input in (False, True):
        analyze_excitatory_with_static_inhibition(
            model,
            short_states,
            initial_state_traj=state_traj,
            trial_inputs=short_inputs,
            include_external_input=include_external_input,
        )
        analyze_inhibitory_with_static_excitation(
            model,
            short_states,
            initial_state_traj=state_traj,
            trial_inputs=short_inputs,
            include_external_input=include_external_input,
        )

    for region in ("exc", "inhib"):
        analyze_isolated_region(model, state_traj, region)

    plot_trials(data, outputs)
    plot_region_pca(model, state_traj)
    unique_fps, _ = find_fixed_points(model, state_traj)
    fp_states = unique_fps.xstar
    plot_fixed_points(state_traj, fp_states)
    analyze_nonlinear_flow(model, state_traj, fp_states)
    stability, eigenvectors = analyze_stability(model, state_traj, fp_states)
    if unique_fps.n:
        flow_finder = analyze_linear_flow(model, state_traj, fp_states, stability)
    else:
        print("No fixed points found; skipping local flow and alignment plots.")


if __name__ == "__main__":
    main()
