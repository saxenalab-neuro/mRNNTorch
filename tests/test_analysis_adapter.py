"""The shared adapter preserves each model's state dynamics and gradients."""

import pytest
import torch

from mrnntorch import ElmanmRNN, mRNN
from mrnntorch.analysis import mRNNAdapter


@pytest.mark.parametrize("model_type", [mRNN, ElmanmRNN])
@pytest.mark.parametrize("batch_first", [True, False])
def test_adapter_dynamics(model_type, batch_first):
    rnn = model_type(
        device="cpu", activation="tanh", batch_first=batch_first,
        rec_constrained=False, inp_constrained=False,
    )
    rnn.add_recurrent_region("r", num_units=2, sign="pos", base_firing=0, init=0)
    rnn.add_input_region("i", num_units=1, sign="pos")
    rnn.add_recurrent_connection("r", "r")
    rnn.add_input_connection("i", "r")
    rnn.finalize_connectivity()
    parameter_flags = [parameter.requires_grad for parameter in rnn.parameters()]
    adapter = mRNNAdapter(rnn)
    state = torch.tensor([[0.4, -0.2]], requires_grad=True)
    inp = torch.tensor([[0.3]], requires_grad=True)
    stim = torch.tensor([[0.1, -0.1]])
    weights = torch.tensor([[0.2, -0.3], [0.5, 0.1]])

    hidden = torch.tanh(state) if model_type is mRNN else state
    drive = hidden @ weights.T + inp @ (rnn.W_inp * rnn.W_inp_mask).T
    drive = drive + rnn.tonic_inp + stim
    expected = (
        state + rnn.alpha * (-state + drive)
        if model_type is mRNN else torch.tanh(drive)
    )
    actual = adapter.step(inp, state, stim_input=stim, W_rec=weights)
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.sum(), (state, inp), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (state, inp))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)

    torch.testing.assert_close(adapter.activity(state), hidden)
    initial = rnn.batched_initial_condition(3)
    torch.testing.assert_close(
        adapter.batched_initial_condition(3),
        initial[0] if model_type is mRNN else initial,
    )

    time_dim = 1 if batch_first else 0
    sequence = torch.stack([inp, inp], dim=time_dim)
    native = rnn(sequence, state)
    torch.testing.assert_close(
        adapter(sequence, state), native[0] if model_type is mRNN else native
    )
    assert [parameter.requires_grad for parameter in rnn.parameters()] == parameter_flags


def test_rejects_unsupported_model():
    with pytest.raises(TypeError, match="mRNN"):
        mRNNAdapter(torch.nn.RNN(1, 2))


@pytest.mark.parametrize("batch_first", [True, False])
def test_independent_leaky_activity_jacobian(batch_first):
    rnn = mRNN(device="cpu", activation="tanh", batch_first=batch_first)
    rnn.add_recurrent_region("r", num_units=2, sign="pos", base_firing=0, init=0)
    rnn.add_input_region("i", num_units=1, sign="pos")
    rnn.add_recurrent_connection("r", "r")
    rnn.add_input_connection("i", "r")
    rnn.finalize_connectivity()
    adapter = mRNNAdapter(rnn)
    inp = torch.tensor([[0.3]])
    x = torch.tensor([[0.4, -0.2]])
    h = torch.tensor([[0.6, 0.1]])  # Deliberately different from tanh(x).
    weights = torch.tensor([[0.2, -0.3], [0.5, 0.1]])

    def activity_step(activity):
        return adapter.step(inp, x, h=activity, W_rec=weights)

    h_next = activity_step(h)
    jac = torch.autograd.functional.jacobian(activity_step, h).reshape(2, 2)
    expected = torch.diag(1 - h_next[0].square()) @ (rnn.alpha * weights)
    torch.testing.assert_close(jac, expected)

    time_dim = 1 if batch_first else 0
    sequence = torch.stack([inp, inp], dim=time_dim)
    native_x, native_h = rnn(sequence, x, h0=h, W_rec=weights)
    states = adapter(sequence, x, h=h, W_rec=weights)
    torch.testing.assert_close(states, native_h)

    with pytest.raises(ValueError, match="same"):
        adapter.step(inp, x, h=h[:, :1])


def test_elman_rejects_separate_activity():
    adapter = mRNNAdapter(ElmanmRNN(device="cpu"))
    with pytest.raises(ValueError, match="Elman"):
        adapter.step(torch.zeros(1, 1), torch.zeros(1, 2), h=torch.zeros(1, 2))
