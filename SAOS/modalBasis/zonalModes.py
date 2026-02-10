import numpy as np
import torch

def generate_zonal_modes(dm, nModes=None, useTorch=False):

    # If the modes are not specified, use the number of actuators --> maximum number of modes
    if nModes is None:
        nModes = dm.nValidAct
    
    # Assign each actuator to a layer of the zonal modes matrix

    identity = np.eye(dm.coordinates.shape[0])

    zonal_modes = identity[:, dm.validAct]

    # Check torch option
    if useTorch:
        return torch.tensor(zonal_modes, dtype=torch.float32)

    return zonal_modes