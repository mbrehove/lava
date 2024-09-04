import typing as ty
from enum import IntEnum, unique
import numpy as np
import os
from typing import Any, Dict

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel

try:
    from lava.magma.core.model.nc.ports import NcInPort, NcOutPort
    from lava.magma.core.model.nc.type import LavaNcType
    from lava.magma.core.model.nc.var import NcVar
    from lava.magma.core.model.nc.net import NetL2
    from lava.magma.core.model.nc.tables import Nodes
    from lava.magma.core.model.nc.model import AbstractNcProcessModel

    loihi_available = True
except ImportError:
    print("Could not import NC model. Default to using CPU implementation")
    loihi_available = False

from lava.magma.core.resources import Loihi2NeuroCore
from lava.magma.core.decorator import implements, requires


@unique
class ActivationMode(IntEnum):
    """Enum for synapse sigma delta activation mode. Options are
    UNIT: 0
    RELU: 1
    """

    UNIT = 0
    RELU = 1


class QANN(AbstractProcess):
    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        scale: int,
        bias: ty.Optional[int] = 0,
        act_mode: ty.Optional[ActivationMode] = ActivationMode.RELU,
        # cum_error: ty.Optional[bool] = False,
        bias_exp: ty.Optional[int] = 0,
        scale_exp: ty.Optional[int] = 0,
        threshold: ty.Optional[int] = 0,
    ) -> None:
        super().__init__(
            shape=shape,
            scale=scale,
            bias=bias,
            act_mode=act_mode,
            # cum_error=cum_error,
            bias_exp=bias_exp,
            scale_exp=scale_exp,
            threshold=threshold,
        )

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)

        if type(scale) == int:
            self.scale = Var(shape=(1,), init=scale)
        elif type(scale) == np.ndarray:
            self.scale = Var(shape=shape, init=scale)
        elif np.isscalar(scale) and np.issubdtype(type(scale), np.integer):
            self.scale = Var(shape=(1,), init=scale)
        else:
            raise ValueError("scale must be an int or np.ndarray")
        self.sigma = Var(shape=shape, init=0)
        self.act = Var(shape=shape, init=0)
        self.residue = Var(shape=shape, init=0)
        self.error = Var(shape=shape, init=0)
        self.bias = Var(shape=shape, init=bias)
        self.bias_exp = Var(shape=(1,), init=bias_exp)
        self.scale_exp = Var(shape=(1,), init=scale_exp)
        self.threshold = Var(shape=(1,), init=threshold)
        # self.cum_error = Var(shape=(1,), init=cum_error)

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        """Return shape of the Process."""
        return self.proc_params["shape"]


if loihi_available:

    def coerce_to_shape(var, shape):
        """Coerce a variable to a given shape. Tile over channels if necessary."""
        if np.isscalar(var):
            return var
        if var.shape == shape:
            return var
        if np.prod(var.shape) == np.prod(shape):
            return var.reshape(shape)
        if len(shape) == 3:
            if np.prod(var.shape) == shape[2]:
                return np.tile(var.reshape(1, 1, -1), (shape[0], shape[1], 1))
        else:
            raise ValueError(f"Cannot coerce {var.shape} to {shape}")

    @implements(proc=QANN, protocol=LoihiProtocol)
    @requires(Loihi2NeuroCore)
    class NcModelQANN(AbstractNcProcessModel):
        """Implementation of a sigma-delta neural (SDN) process model using
        the micro-coded (ucoded) description on Loihi 2.
        """

        # Declare port implementation
        a_in: NcInPort = LavaNcType(NcInPort, np.int32, precision=24)
        s_out: NcOutPort = LavaNcType(NcOutPort, np.int32, precision=24)

        # Declare variable implementation
        scale: NcVar = LavaNcType(NcVar, np.int32, precision=12)
        sigma: NcVar = LavaNcType(NcVar, np.int32, precision=24)
        act: NcVar = LavaNcType(NcVar, np.int32, precision=24)
        residue: NcVar = LavaNcType(NcVar, np.int32, precision=24)
        error: NcVar = LavaNcType(NcVar, np.int32, precision=24)
        bias: NcVar = LavaNcType(NcVar, np.int32, precision=16)

        bias_exp: NcVar = LavaNcType(NcVar, np.int32, precision=3)
        scale_exp: NcVar = LavaNcType(NcVar, np.int32, precision=3)
        # cum_error: NcVar = LavaNcType(NcVar, bool, precision=1)

        def allocate(self, net: NetL2):
            """Allocates neural resources in neuro core."""
            shape = self.proc_params["shape"]
            flat_shape = (np.prod(shape),)
            num_message_bits = self.proc_params.get("num_message_bits", 16)

            # Allocate neurons
            curr_dir = os.path.dirname(os.path.realpath(__file__))

            scale_exp = self.scale_exp.var.get()
            bias_exp = self.bias_exp.var.get()
            bias = self.bias.var.get()
            scale = self.scale.var.get()
            bias = coerce_to_shape(bias, shape)
            scale = coerce_to_shape(scale, shape)
            act_ref = self.residue.var.get() - self.act.var.get()
            if np.isscalar(
                scale
            ):  # Compile the scale into the code if its a scalar
                ucode_file = os.path.join(curr_dir, "qann.dasm")
                neurons_cfg: Nodes = net.neurons_cfg.allocate_ucode(
                    shape=(1,),
                    ucode=ucode_file,
                    scale_exp=scale_exp,
                    bias_exp=bias_exp,
                    scale=scale,
                )
                neurons: Nodes = net.neurons.allocate_ucode(
                    shape=flat_shape,
                    sigma=self.sigma,
                    act_ref=act_ref,
                    bias=bias,
                )
            else:
                raise ("Channel wise scaling not implemented yet")
                ucode_file = os.path.join(curr_dir, "qann_channel_wise.dasm")
                neurons_cfg: Nodes = net.neurons_cfg.allocate_ucode(
                    shape=(1,),
                    ucode=ucode_file,
                    scale_exp=scale_exp,
                    bias_exp=bias_exp,
                )
                neurons: Nodes = net.neurons.allocate_ucode(
                    shape=flat_shape,
                    sigma=self.sigma,
                    act_ref=act_ref,
                    bias=bias,
                    scale=scale,
                )
            # Allocate output axons
            ax_out: Nodes = net.ax_out.allocate(
                shape=flat_shape, num_message_bits=num_message_bits
            )

            # Connect InPort of Process to neurons
            self.a_in.connect(neurons)
            # Connect Nodes
            neurons.connect(neurons_cfg)
            neurons.connect(ax_out)
            # Connect output axon to OutPort of Process
            ax_out.connect(self.s_out)


@implements(proc=QANN, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyQANNModelFixed(PyLoihiProcessModel):
    """Fixed point implementation of Sigma Delta neuron."""

    a_in = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    s_out = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)

    # Declare variable implementation
    scale: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=12)
    sigma: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    act: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    residue: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    error: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    bias: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=16)

    bias_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    scale_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    threshold: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=8)
    # cum_error: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)

    def run_spk(self) -> None:
        # Receive synaptic input
        a_in_data = self.a_in.recv()

        # Sigma dynamics
        self.sigma = self.sigma + a_in_data

        if np.max(np.abs(self.sigma)) > 2**23 - 1:
            print(f"Overflow error max sigma: {np.max(np.abs(self.sigma))}")
            # raise ValueError("Overflow error")

        # Activation dynamics with bit shift
        act = (self.bias << self.bias_exp) + self.sigma
        act = np.maximum(0, act)
        act = (self.scale * act) >> 12
        act = np.right_shift(act, self.scale_exp)

        # Delta dynamics
        delta = act - self.act + self.residue
        s_out = np.where(
            np.abs(delta) > self.threshold,
            delta,
            0,
        )
        if np.max(np.abs(delta)) > 2**7 - 1:
            print(f"Overflow error max act: {np.max(np.abs(delta))}")
            # raise ValueError("Overflow error")
        self.residue = delta - s_out
        self.act = act

        # Send spike output
        self.s_out.send(s_out)


# class QANNSigma(AbstractProcess):
#     def __init__(self, *, shape: ty.Tuple[int, ...]) -> None:
#         """Sigma integration unit process definition. A sigma process is simply
#         a cumulative accumulator over time.

#         Sigma dynamics:
#         sigma = a_in + sigma                      # sigma dendrite
#         a_out = sigma

#         Parameters
#         ----------
#         shape: Tuple
#             shape of the sigma process. Default is (1,).
#         """
#         super().__init__(shape=shape)

#         self.a_in = InPort(shape=shape)
#         self.s_out = OutPort(shape=shape)

#         self.sigma = Var(shape=shape, init=0)

#     @property
#     def shape(self) -> ty.Tuple[int, ...]:
#         """Return shape of the Process."""
#         return self.proc_params["shape"]


# def as_signed(arr, num_bits):
#     if num_bits <= 0 or num_bits >= 32:
#         raise ValueError("Number of bits should be between 1 and 31.")

#     # Calculate the bitmask for the desired number of bits
#     bitmask = (1 << num_bits) - 1

#     # Convert to 32-bit signed integer
#     arr_32bit = np.int32(arr)

#     # Truncate to the desired number of bits
#     arr_truncated = np.bitwise_and(arr_32bit, bitmask)

#     # Handle negative values by applying two's complement
#     negative_mask = np.bitwise_and(arr_truncated, 1 << (num_bits - 1))
#     arr_signed = np.where(
#         negative_mask > 0, arr_truncated - (1 << num_bits), arr_truncated
#     )

#     return arr_signed.astype(np.int32)


# @implements(proc=QANNSigma, protocol=LoihiProtocol)
# @requires(CPU)
# @tag("fixed_pt")
# class QANNSigmaModel(PyLoihiProcessModel):
#     """Fixed point implementation of Sigma decoding"""

#     # a_in = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
#     # s_out = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
#     # sigma: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
#     a_in = LavaPyType(PyInPort.VEC_DENSE, float)
#     s_out = LavaPyType(PyOutPort.VEC_DENSE, float)
#     sigma: np.ndarray = LavaPyType(np.ndarray, float)

#     def run_spk(self) -> None:
#         raw_data = self.a_in.recv()
#         # if not (raw_data.max() < 2**8):
#         #     print(f"max: {raw_data.max()}")
#         # if not (raw_data.min() >= -(2**8)):
#         #     print(f"min: {raw_data.min()}")
#         if raw_data.max() > 2**16:
#             print(f"Overflow error: max: {raw_data.max()}")
#         data = as_signed(raw_data, 16)
#         self.sigma = data + self.sigma
#         self.s_out.send(self.sigma)
