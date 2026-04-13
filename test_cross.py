import numpy as np
import torch
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'OOPAO')))

from SAOS.Telescope import Telescope
from SAOS.DeformableMirror import DeformableMirror

diameter = 4.2 
obs_diameter = 0.0 
sampling_time = 1/2000. 
n_subaperture_red = 36
resolution = 216
pixel_size = diameter / resolution
tel_fov = 0

est_tel = Telescope(diameter=diameter,
                    resolution=resolution,
                    centralObstruction=obs_diameter/diameter,
                    samplingTime=sampling_time,
                    fov=tel_fov)

asm = DeformableMirror(telescope=est_tel,
                        nActs=n_subaperture_red+1,
                        altitude=0,
                        typeDM='cartesian',
                        pitch=diameter/n_subaperture_red,
                        mechCoupling=0.60,
                        **{'validActThreshpercentage': 0.7533})

act_coords = asm.coordinates[asm.validAct]
tilt_amp = 1e-6
commands_1d = (act_coords[:, 0] / (diameter/2)) * tilt_amp
commands_1d_torch = torch.tensor(commands_1d, dtype=torch.float64, device=asm.device)

# 1. get_desired_opd
opd_desired_2d = asm.get_desired_opd(commands_1d_torch, asm.interp_matrix) * asm.dm_layer.metapupil
opd_desired_1d = opd_desired_2d.reshape(-1)

# 2. Projection
coefs_corrected = asm.projector @ opd_desired_1d

# 3. Final surface
opd_highres = asm.influenceFunctions @ coefs_corrected
opd_highres_2d = opd_highres.reshape(resolution, resolution).cpu().numpy() * asm.dm_layer.metapupil

# Save a 1D cross-section at y=0 (middle row)
mid = resolution // 2
x_axis = np.linspace(-diameter/2, diameter/2, resolution)

plt.figure(figsize=(10, 6))
plt.plot(x_axis, opd_desired_2d.cpu().numpy()[mid, :], label='Desired OPD (Linear Interp)', linestyle='--')
plt.plot(x_axis, opd_highres_2d[mid, :], label='Fitted OPD (Projector -> Gaussians)', linewidth=2)

# Overlay valid actuators
act_x = act_coords[np.abs(act_coords[:, 1]) < asm.pitch/2, 0]
act_z = (act_x / (diameter/2)) * tilt_amp
plt.scatter(act_x, act_z, color='red', zorder=5, label='Actuator Commands')

plt.axvline(-diameter/2, color='k', linestyle=':', label='Pupil Edge')
plt.axvline(diameter/2, color='k', linestyle=':')

plt.legend()
plt.title('1D Cross-section of OPD fitting')
plt.grid()
plt.savefig('debug_cross_section.png')
