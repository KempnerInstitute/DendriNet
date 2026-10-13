"""Endpoint-normalized compartment activation for voltages in [0, 1]."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class EndpointNormalizedTanh(nn.Module):
    """Learn ``m > 0, b`` in (tanh(mV-b)-tanh(-b))/(tanh(m-b)-tanh(-b)).

    Here b is an offset, not the voltage midpoint used by ParametricTanh.
    The exponential identity avoids subtracting two saturated tanh values.
    Half-precision inputs are evaluated in float32 and cast back afterwards.
    This activation is qualified for ordinary backpropagation on shunting
    voltages, with fixed initial m and b (no occupancy calibration).
    """

    # Data-driven reactivation calibration writes a voltage midpoint, which is
    # not this activation's offset b; calibration requests fail closed.
    supports_midpoint_calibration = False

    def __init__(self, output_dim, init_m=3.0, init_b=1.5):
        super().__init__()
        self.log_m = nn.Parameter(torch.empty(output_dim))
        self.b = nn.Parameter(torch.empty(output_dim))
        self.initialize(init_m, init_b)

    @torch.no_grad()
    def initialize(self, slope_val, mid_val):
        if not math.isfinite(slope_val) or slope_val <= 0 or not math.isfinite(mid_val):
            raise ValueError("endpoint_tanh requires finite positive m and finite b")
        self.log_m.fill_(math.log(slope_val))
        self.b.fill_(mid_val)

    def forward(self, voltage):
        value = (
            voltage.float()
            if voltage.dtype in (torch.float16, torch.bfloat16)
            else voltage
        )
        m, b = self.log_m.exp(), self.b
        ratio = torch.expm1(-2 * m * value) / torch.expm1(-2 * m)
        exponent = (
            2 * m * (value - 1)
            + F.softplus(2 * (m - b))
            - F.softplus(2 * (m * value - b))
        )
        return (ratio * exponent.exp()).to(voltage.dtype)
