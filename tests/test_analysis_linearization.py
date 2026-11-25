"""Unit tests for the Linearization analysis utilities.

These tests focus on derivative helpers and Jacobian/EVD behavior while
keeping network construction minimal.
"""

import pytest
import torch

from mrnntorch.analysis.linear import mLinearization
from mrnntorch import mRNN, ElmanmRNN


def _build_leaky_mrnn(activation="relu") -> mRNN:
    """Construct a minimal mRNN with two recurrent regions on CPU."""
    mrnn = mRNN(
        device="cpu",
        rec_constrained=False,
        inp_constrained=False,
        activation=activation,
    )
    mrnn.add_recurrent_region(name="r1", num_units=2, sign="pos", base_firing=0, init=0)
    mrnn.add_recurrent_region(name="r2", num_units=1, sign="pos", base_firing=0, init=0)
    return mrnn


def _build_leaky_mrnn_with_inputs(activation="relu") -> mRNN:
    """Construct a minimal mRNN with input connectivity for flow calls."""
    mrnn = _build_leaky_mrnn(activation=activation)
    mrnn.add_input_region(name="i1", num_units=1, sign="pos")
    mrnn.add_recurrent_connection("r1", "r1")
    mrnn.add_recurrent_connection("r1", "r2")
    mrnn.add_recurrent_connection("r2", "r1")
    mrnn.add_recurrent_connection("r2", "r2")
    mrnn.add_input_connection("i1", "r1")
    mrnn.add_input_connection("i1", "r2")
    mrnn.finalize_connectivity()
    return mrnn


def _build_elman_mrnn(activation="relu") -> ElmanmRNN:
    """Construct a minimal mRNN with two recurrent regions on CPU."""
    mrnn = ElmanmRNN(
        device="cpu",
        rec_constrained=False,
        inp_constrained=False,
        activation=activation,
    )
    mrnn.add_recurrent_region(name="r1", num_units=2, sign="pos", base_firing=0, init=0)
    mrnn.add_recurrent_region(name="r2", num_units=1, sign="pos", base_firing=0, init=0)
    return mrnn


def _build_elman_mrnn_with_inputs(activation="relu") -> ElmanmRNN:
    """Construct a minimal mRNN with input connectivity for flow calls."""
    mrnn = _build_elman_mrnn(activation=activation)
    mrnn.add_input_region(name="i1", num_units=1, sign="pos")
    mrnn.add_recurrent_connection("r1", "r1")
    mrnn.add_recurrent_connection("r1", "r2")
    mrnn.add_recurrent_connection("r2", "r1")
    mrnn.add_recurrent_connection("r2", "r2")
    mrnn.add_input_connection("i1", "r1")
    mrnn.add_input_connection("i1", "r2")
    mrnn.finalize_connectivity()
    return mrnn


# -------------- Testing leaky mRNN --------------------


def test_jacobian_matches_weight_x_l():
    """Linear activation yields the unscaled leaky update Jacobian by default."""
    mrnn = _build_leaky_mrnn_with_inputs(activation="relu")
    x = torch.ones(3)
    inp = torch.zeros(1)
    lin = mLinearization(mrnn, inp, x)
    jac, jac_inp = lin.jacobian, lin.jacobian_inp
    assert torch.allclose(
        jac, (1 - mrnn.alpha) * torch.eye(3) + mrnn.alpha * mrnn.W_rec
    )
    assert torch.allclose(jac_inp, mrnn.alpha * mrnn.W_inp)


def test_jacobian_matches_weight_x_r1_l():
    """Linear activation yields the unscaled leaky update Jacobian by default."""
    mrnn = _build_leaky_mrnn_with_inputs(activation="relu")
    x = torch.ones(3)
    inp = torch.zeros(1)
    lin = mLinearization(mrnn, inp, x, "r1")
    jac, jac_inp = lin.jacobian, lin.jacobian_inp
    assert torch.allclose(
        jac,
        mrnn.get_weight_subset(
            "r1", W=(1 - mrnn.alpha) * torch.eye(3) + mrnn.alpha * mrnn.W_rec
        ),
    )
    assert torch.allclose(jac_inp, (mrnn.alpha * mrnn.W_inp)[:2, :])


def test_jacobian_matches_weight_x_r2_l():
    """Linear activation yields the unscaled leaky update Jacobian by default."""
    mrnn = _build_leaky_mrnn_with_inputs(activation="relu")
    x = torch.ones(3)
    inp = torch.zeros(1)
    lin = mLinearization(mrnn, inp, x, "r2")
    jac, jac_inp = lin.jacobian, lin.jacobian_inp
    assert torch.allclose(
        jac,
        mrnn.get_weight_subset(
            "r2", W=(1 - mrnn.alpha) * torch.eye(3) + mrnn.alpha * mrnn.W_rec
        ),
    )
    assert torch.allclose(jac_inp, mrnn.alpha * mrnn.W_inp[2:, :])


def test_jacobian_matches_weight_h_l():
    """Linear activation yields the unscaled leaky update Jacobian by default."""
    mrnn = _build_leaky_mrnn_with_inputs(activation="relu")
    x = torch.tensor([1.0, 1.0, -1.0])
    h = torch.tensor([1.0, 1.0, 0.0])
    inp = torch.zeros(1)
    x_next, _ = mrnn(inp.unsqueeze(0).unsqueeze(0), x.unsqueeze(0))
    x_next = x_next.squeeze()
    dx = torch.where(x_next > 0, 1.0, 0.0)
    lin = mLinearization(mrnn, inp, x, dh=True)
    jac, jac_inp = lin.jacobian, lin.jacobian_inp
    assert torch.allclose(jac, torch.diag(dx) @ (mrnn.W_rec))
    assert torch.allclose(jac_inp, torch.diag(dx) @ (mrnn.W_inp))


def test_jacobian_matches_weight_h_r1_l():
    """Linear activation yields the unscaled leaky update Jacobian by default."""
    mrnn = _build_leaky_mrnn_with_inputs(activation="relu")
    x = torch.tensor([1.0, 1.0, -1.0])
    h = torch.tensor([1.0, 1.0, 0.0])
    inp = torch.zeros(1)
    x_next, _ = mrnn(inp.unsqueeze(0).unsqueeze(0), x.unsqueeze(0))
    x_next = x_next.squeeze()
    dx = torch.where(x_next > 0, 1.0, 0.0)
    lin = mLinearization(mrnn, inp, x, "r1", dh=True)
    jac, jac_inp = lin.jacobian, lin.jacobian_inp
    assert torch.allclose(
        jac,
        mrnn.get_weight_subset("r1", W=torch.diag(dx) @ (mrnn.W_rec)),
    )
    assert torch.allclose(jac_inp, (torch.diag(dx) @ (mrnn.W_inp))[:2, :])


def test_jacobian_matches_weight_h_r2_l():
    """alpha_scaling=True scales out the leaky update alpha."""
    mrnn = _build_leaky_mrnn_with_inputs(activation="relu")
    x = torch.tensor([1.0, 1.0, -1.0])
    h = torch.tensor([1.0, 1.0, 0.0])
    inp = torch.zeros(1)
    x_next, _ = mrnn(inp.unsqueeze(0).unsqueeze(0), x.unsqueeze(0))
    x_next = x_next.squeeze()
    dx = torch.where(x_next > 0, 1.0, 0.0)
    lin = mLinearization(mrnn, inp, x, "r2", dh=True, alpha_scaling=True)
    jac, jac_inp = lin.jacobian, lin.jacobian_inp
    assert torch.allclose(
        jac,
        mrnn.get_weight_subset("r2", W=torch.diag(dx) @ (mrnn.alpha * mrnn.W_rec)),
    )
    assert torch.allclose(jac_inp, (torch.diag(dx) @ (mrnn.alpha * mrnn.W_inp))[2:, :])


def test_eigendecomposition_returns_real_imag_parts_l():
    """Eigen decomposition returns real/imag parts and eigenvectors."""
    mrnn = _build_leaky_mrnn_with_inputs()
    x = torch.zeros(3)
    lin = mLinearization(mrnn, torch.zeros(1), x)
    reals, ims, vecs = (
        lin.jac_rec_real_eigs, lin.jac_rec_im_eigs, lin.jac_rec_eigvecs
    )
    # Eigenvectors are returned column-wise for a square matrix.
    assert vecs.shape == (3, 3)
    assert len(reals) == 3
    assert len(ims) == 3


def test_jacobian_requires_1d_x_l():
    """Construction asserts on non-1D inputs to avoid shape ambiguity."""
    mrnn = _build_leaky_mrnn_with_inputs()
    with pytest.raises(AssertionError):
        mLinearization(mrnn, torch.zeros(1, 1), torch.zeros(1, 2))


# -------------- Testing elman mRNN --------------------


def test_linear_jacobian_matches_weight_e():
    """Linear activation yields Jacobian equal to W_rec (scaled by alpha)."""
    mrnn = _build_elman_mrnn_with_inputs(activation="linear")
    x = torch.zeros(3)
    inp = torch.zeros(1)
    lin = mLinearization(mrnn, inp, x)
    jac, jac_inp = lin.jacobian, lin.jacobian_inp
    assert torch.allclose(jac, mrnn.W_rec)
    assert torch.allclose(jac_inp, mrnn.W_inp)


def test_jacobian_matches_weight_r1_e():
    """Linear activation yields Jacobian equal to W_rec (scaled by alpha)."""
    mrnn = _build_elman_mrnn_with_inputs(activation="relu")
    h = torch.tensor([1.0, 1.0, -1.0])
    inp = torch.zeros(1)
    h_next = mrnn(inp.unsqueeze(0).unsqueeze(0), h.unsqueeze(0))
    h_next = h_next.squeeze()
    dh = torch.where(h_next > 0, 1.0, 0.0)
    lin = mLinearization(mrnn, inp, h, "r1")
    jac, jac_inp = lin.jacobian, lin.jacobian_inp
    assert torch.allclose(
        jac, mrnn.get_weight_subset("r1", W=torch.diag(dh) @ mrnn.W_rec)
    )
    assert torch.allclose(jac_inp, (torch.diag(dh) @ (mrnn.W_inp))[:2, :])


def test_jacobian_matches_weight_r2_e():
    """Linear activation yields Jacobian equal to W_rec (scaled by alpha)."""
    mrnn = _build_elman_mrnn_with_inputs(activation="relu")
    h = torch.tensor([1.0, 1.0, -1.0])
    inp = torch.zeros(1)
    h_next = mrnn(inp.unsqueeze(0).unsqueeze(0), h.unsqueeze(0))
    h_next = h_next.squeeze()
    dh = torch.where(h_next > 0, 1.0, 0.0)
    lin = mLinearization(mrnn, inp, h, "r2")
    jac, jac_inp = lin.jacobian, lin.jacobian_inp
    assert torch.allclose(
        jac, mrnn.get_weight_subset("r2", W=torch.diag(dh) @ mrnn.W_rec)
    )
    assert torch.allclose(jac_inp, (torch.diag(dh) @ mrnn.W_inp)[2:, :])


def test_eigendecomposition_returns_real_imag_parts_e():
    """Eigen decomposition returns real/imag parts and eigenvectors."""
    mrnn = _build_elman_mrnn_with_inputs()
    x = torch.zeros(3)
    lin = mLinearization(mrnn, torch.zeros(1), x)
    reals, ims, vecs = (
        lin.jac_rec_real_eigs, lin.jac_rec_im_eigs, lin.jac_rec_eigvecs
    )
    # Eigenvectors are returned column-wise for a square matrix.
    assert vecs.shape == (3, 3)
    assert len(reals) == 3
    assert len(ims) == 3


def test_jacobian_requires_1d_x_e():
    """Construction asserts on non-1D inputs to avoid shape ambiguity."""
    mrnn = _build_elman_mrnn_with_inputs()
    with pytest.raises(AssertionError):
        mLinearization(mrnn, torch.zeros(1, 1), torch.zeros(1, 2))


@pytest.mark.parametrize("model_type", [mRNN, ElmanmRNN])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("activity_mode", [True, False])
@pytest.mark.parametrize("stimulated", [True, False])
@pytest.mark.parametrize("x_is_h", [True, False])
def test_shared_linearization_matches_native_jacobian(model_type, batch_first, activity_mode, stimulated, x_is_h):
    builder = (
        _build_leaky_mrnn_with_inputs if model_type is mRNN
        else _build_elman_mrnn_with_inputs
    )
    rnn = builder(activation="tanh")
    rnn.batch_first = batch_first
    state = torch.tensor([0.4, -0.2, 0.1])
    inp = torch.tensor([0.3])
    dh = activity_mode and model_type is mRNN
    x_is_h = x_is_h and dh
    h = rnn.activation(state) if dh else None
    stim = torch.tensor([0.7, -0.8, 0.5]) if stimulated else None
    native_stim = None if stim is None else stim.reshape(1, 1, -1)

    def native_update(u, variable):
        if model_type is mRNN:
            xs, hs = rnn(
                u.reshape(1, 1, -1),
                (state if dh else variable).unsqueeze(0),
                h0=variable.unsqueeze(0) if dh else None,
                stim_input=native_stim,
            )
            return (hs if dh else xs).reshape(-1)
        return rnn(
            u.reshape(1, 1, -1), variable.unsqueeze(0), stim_input=native_stim
        ).reshape(-1)

    variable = h if dh else state
    forward_variable = state if x_is_h else variable
    expected_input, expected_state = torch.autograd.functional.jacobian(
        native_update, (inp, variable)
    )
    lin = mLinearization(rnn, inp, state, stim_input=stim, dh=dh, alpha_scaling=True)
    jac, jac_input = lin.jacobian, lin.jacobian_inp
    torch.testing.assert_close(jac, expected_state)
    torch.testing.assert_close(jac_input, expected_input)
    torch.testing.assert_close(
        lin(torch.zeros_like(inp), torch.zeros_like(state), x_is_h=x_is_h),
        native_update(inp, forward_variable),
    )
    subset = mLinearization(
        rnn, inp, state, "r1", stim_input=stim, dh=dh, alpha_scaling=True
    )
    excluded = subset.jacobian_exc
    torch.testing.assert_close(excluded, expected_state[:2, 2:])
    delta = torch.tensor([0.01, -0.02, 0.03])
    delta_input = torch.tensor([0.02])
    torch.testing.assert_close(
        subset.forward(
            delta_input, delta[:2], delta_state_static=delta[2:],
            x_is_h=x_is_h,
        ),
        (native_update(inp, forward_variable) + expected_state @ delta
         + expected_input @ delta_input)[:2],
    )
    reals, imags, vectors = (
        lin.jac_rec_real_eigs, lin.jac_rec_im_eigs, lin.jac_rec_eigvecs
    )
    torch.testing.assert_close(
        expected_state.to(vectors.dtype) @ vectors,
        vectors * torch.complex(reals, imags),
    )
    if dh:
        normalized_lin = mLinearization(rnn, inp, state, stim_input=stim, dh=True)
        normalized, normalized_input = normalized_lin.jacobian, normalized_lin.jacobian_inp
        torch.testing.assert_close(normalized, expected_state / rnn.alpha)
        torch.testing.assert_close(normalized_input, expected_input / rnn.alpha)
