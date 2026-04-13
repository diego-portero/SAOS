import numpy as np
import torch
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'OOPAO')))

from SAOS.Telescope import Telescope
from SAOS.DeformableMirror import DeformableMirror

test_logger = None

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

# Let's generate a smooth 1D tilt command evaluated at actuator coordinates
act_coords = asm.coordinates[asm.validAct]
tilt_amp = 1e-6
commands_1d = (act_coords[:, 0] / (diameter/2)) * tilt_amp
commands_1d_torch = torch.tensor(commands_1d, dtype=torch.float64, device=asm.device)

# Emulate what updateDMShape is doing internally:
# 1. get_desired_opd (Voronoi pixelation)
opd_desired_2d = asm.get_desired_opd(commands_1d_torch, asm.interp_matrix) * asm.dm_layer.metapupil
opd_desired_1d = opd_desired_2d.reshape(-1)

# 2. Projection
coefs_corrected = asm.projector @ opd_desired_1d

# 3. Final surface
opd_highres = asm.influenceFunctions @ coefs_corrected
opd_highres_2d = opd_highres.reshape(resolution, resolution).cpu().numpy() * asm.dm_layer.metapupil

# Plotting the diagnostic
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

im0 = axs[0].imshow(opd_desired_2d.cpu().numpy() * 1e9, origin='lower', cmap='seismic')
axs[0].set_title('1. "Desired OPD" (Voronoi cells) [nm]')
plt.colorbar(im0, ax=axs[0])

# Plotting the coefficients
axs[1].scatter(act_coords[:, 0], act_coords[:, 1], c=coefs_corrected.cpu().numpy(), cmap='seismic')
axs[1].set_title('2. Projected Coefficients')
axs[1].axis('equal')

im2 = axs[2].imshow(opd_highres_2d * 1e9, origin='lower', cmap='seismic')
axs[2].set_title('3. Final DM Surface [nm] -> RINGS!')
plt.colorbar(im2, ax=axs[2])

plt.suptitle('Diagnostic: Why the Nearest Neighbor (Voronoi) mapping causes rings')
plt.tight_layout()
plt.savefig('debug_rings.png')

print("OPD PtV [nm]:", (opd_highres_2d.max() - opd_highres_2d.min()) * 1e9)
print("Diagnostic image saved as debug_rings.png")
