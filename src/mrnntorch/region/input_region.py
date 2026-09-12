"""Input-region definitions for external channels entering an mRNN."""

from mrnntorch.region.region_base import Region, DEFAULT_REGION_BASE


class InputRegion(Region):
    """Input-side region metadata used to assemble input connectivity.

    ``InputRegion`` stores the number of external input channels, output sign,
    device placement, and per-connection parameters inherited from
    :class:`mrnntorch.region.region_base.Region`. Instances are usually created
    through :meth:`mrnntorch.mrnn.mrnn_base.mRNNBase.add_input_region`.
    """

    def __init__(
        self,
        num_units,
        sign=DEFAULT_REGION_BASE["sign"],
        device=DEFAULT_REGION_BASE["device"],
    ):
        """Initialize an input region.

        Args:
            num_units (int): Number of input channels.
            sign (str): "pos" or "neg" indicating sign mask for inputs.
            device (str): Torch device string.
        """
        # Implements base region class
        super(InputRegion, self).__init__(num_units, sign=sign, device=device)
