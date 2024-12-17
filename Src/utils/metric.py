from typing import Sequence

import pytorch_msssim
import torch
import torch.nn.functional as F


class MultiScaleSSIM(Metric):
    input_ranges = ["sigmoid", "tanh"]
    normalization_strategies = [None, "scaling", "clipping"]

    def __init__(
        self,
        output_transform=lambda x: x,
        ms_ssim_kwargs=None,
        normalization_strategy: str = "scaling",
        input_range: str = "sigmoid",
    ):
        self._accumulator = None
        self._count = None

        self._normalization_strategy = normalization_strategy.lower()
        assert (
            self._normalization_strategy in self.normalization_strategies,
            f"Got {self._normalization_strategy} but valid normalization strategies are {self.normalization_strategies}.",
        )

        self._input_range = input_range.lower()
        assert (
            self._input_range in self.input_ranges,
            f"Got {self._input_range} but valid input ranges are {self.input_ranges}.",
        )

        self._ms_ssim_kwargs = {
            "data_range": 1,
            "win_size": 11,
            "win_sigma": 1.5,
            "size_average": False,
            "weights": None,
            "K": (0.01, 0.03),
        }

        if ms_ssim_kwargs:
            self._ms_ssim_kwargs.update(ms_ssim_kwargs)

        super(MultiScaleSSIM, self).__init__(output_transform=output_transform)

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        # Sigmoid input range - [0,1]
        if self._input_range == self.input_ranges[1]:
            x = x + 1
            x = x / 2

        # Min-Max Scaling Normalization
        if self._normalization_strategy == self.normalization_strategies[0]:
            x = x - torch.amin(x, dim=(2, 3, 4), keepdim=True)
            x = x / torch.amax(x, dim=(2, 3, 4), keepdim=True)
        # Clipping Normalization
        elif self._normalization_strategy == self.normalization_strategies[1]:
            x = torch.clamp(x, min=0, max=1)

        return x


class MAE(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._accumulator = None
        self._count = None
        super(MAE, self).__init__(output_transform=output_transform)



class MSE(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._accumulator = None
        self._count = None
        super(MSE, self).__init__(output_transform=output_transform)

