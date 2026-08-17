from typing import Union

import torch
from omegaconf import DictConfig

from .pointmass import ContinuousPointMassModel, DiscretePointMassModel, PointMassModelBase
from .quadrotor import QuadrotorModel
from .velocity_pointmass import VelocityPointMassModel

DYNAMICS_ALIAS = {
    "countinuous_pointmass": ContinuousPointMassModel,
    "discrete_pointmass": DiscretePointMassModel,
    "quadrotor": QuadrotorModel,
    "velocity_pointmass": VelocityPointMassModel
}

def build_dynamics(cfg, device):
    # type: (DictConfig, torch.device) -> Union[ContinuousPointMassModel, DiscretePointMassModel, QuadrotorModel, VelocityPointMassModel]
    return DYNAMICS_ALIAS[cfg.name](cfg, device)