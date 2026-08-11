"""Device placement tests for model tensors and generated outputs."""

import pytest
import torch

from mrnntorch.mrnn.elman_mrnn import ElmanmRNN
from mrnntorch.mrnn.leaky_mrnn import mRNN


DEVICE_PARAMS = [
    pytest.param("cpu", id="cpu"),
    pytest.param(
        "cuda:0",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is not available"
        ),
        id="cuda:0",
    ),
]


def _build_connected_model(model_cls, device):
    model = model_cls(
        device=device,
        activation="tanh",
        rec_constrained=True,
        inp_constrained=True,
    )
    model.add_recurrent_region(
        name="r1",
        num_units=2,
        sign="pos",
        base_firing=0.25,
        init=0.1,
        learnable_bias=True,
    )
    model.add_recurrent_region(
        name="r2",
        num_units=1,
        sign="neg",
        base_firing=0.5,
        init=-0.2,
    )
    model.add_recurrent_connection("r1", "r1")
    model.add_recurrent_connection("r1", "r2")
    model.add_recurrent_connection("r2", "r1")

    model.add_input_region(name="inp", num_units=3, sign="pos")
    model.add_input_connection("inp", "r1")
    model.finalize_connectivity()
    return model


def _iter_region_tensors(model):
    for region_name, region in model.region_dict.items():
        yield f"region_dict.{region_name}.init", region.init
        yield f"region_dict.{region_name}.base_firing", region.base_firing
        for mask_name, mask in region.masks.items():
            yield f"region_dict.{region_name}.masks.{mask_name}", mask
        for connection_name, connection in region.connections.items():
            yield (
                f"region_dict.{region_name}.{connection_name}.parameter",
                connection.parameter,
            )
            yield (
                f"region_dict.{region_name}.{connection_name}.weight_mask",
                connection.weight_mask,
            )
            yield (
                f"region_dict.{region_name}.{connection_name}.sign_matrix",
                connection.sign_matrix,
            )

    for region_name, region in model.inp_dict.items():
        for mask_name, mask in region.masks.items():
            yield f"inp_dict.{region_name}.masks.{mask_name}", mask
        for connection_name, connection in region.connections.items():
            yield (
                f"inp_dict.{region_name}.{connection_name}.parameter",
                connection.parameter,
            )
            yield (
                f"inp_dict.{region_name}.{connection_name}.weight_mask",
                connection.weight_mask,
            )
            yield (
                f"inp_dict.{region_name}.{connection_name}.sign_matrix",
                connection.sign_matrix,
            )


def _iter_model_tensors(model):
    for name, parameter in model.named_parameters():
        yield f"named_parameters.{name}", parameter

    for name in (
        "W_rec",
        "W_rec_mask",
        "W_rec_sign_matrix",
        "W_inp",
        "W_inp_mask",
        "W_inp_sign_matrix",
    ):
        yield name, getattr(model, name)

    for name, mask in model.region_mask_dict.items():
        yield f"region_mask_dict.{name}", mask

    yield from _iter_region_tensors(model)
    yield "initial_condition", model.initial_condition
    yield "tonic_inp", model.tonic_inp


def _assert_tensor_on_device(name, tensor, device):
    assert tensor.device == torch.device(device), f"{name} is on {tensor.device}"


@pytest.mark.parametrize("device", DEVICE_PARAMS)
@pytest.mark.parametrize("model_cls", [mRNN, ElmanmRNN])
def test_model_parameters_masks_and_region_tensors_are_on_device(model_cls, device):
    """Finalized model parameters, masks, and region tensors stay on model device."""
    model = _build_connected_model(model_cls, device)

    for name, tensor in _iter_model_tensors(model):
        _assert_tensor_on_device(name, tensor, device)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_leaky_mrnn_generated_tensors_are_on_device(device):
    """Leaky mRNN initial conditions, noise, stim input path, and outputs use device."""
    model = _build_connected_model(mRNN, device)
    batch_size = 2
    seq_len = 4

    x0, h0 = model.batched_initial_condition(batch_size)
    inp = torch.zeros(batch_size, seq_len, model.total_num_inputs, device=device)
    stim_input = torch.ones(batch_size, seq_len, model.total_num_units, device=device)
    xs, hs = model(inp, x0, h0, stim_input=stim_input, noise=True)

    generated_tensors = {
        "batched_initial_condition.x0": x0,
        "batched_initial_condition.h0": h0,
        "hid_noise": model._hid_noise(batch_size),
        "inp_noise": model._inp_noise(batch_size),
        "forward.xs": xs,
        "forward.hs": hs,
    }
    for name, tensor in generated_tensors.items():
        _assert_tensor_on_device(name, tensor, device)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_elman_mrnn_generated_tensors_are_on_device(device):
    """Elman mRNN initial conditions, noise, stim input path, and outputs use device."""
    model = _build_connected_model(ElmanmRNN, device)
    batch_size = 2
    seq_len = 4

    h0 = model.batched_initial_condition(batch_size)
    inp = torch.zeros(batch_size, seq_len, model.total_num_inputs, device=device)
    stim_input = torch.ones(batch_size, seq_len, model.total_num_units, device=device)
    hs = model(inp, h0, stim_input=stim_input, noise=True)

    generated_tensors = {
        "batched_initial_condition.h0": h0,
        "hid_noise": model._hid_noise(batch_size),
        "inp_noise": model._inp_noise(batch_size),
        "forward.hs": hs,
    }
    for name, tensor in generated_tensors.items():
        _assert_tensor_on_device(name, tensor, device)
