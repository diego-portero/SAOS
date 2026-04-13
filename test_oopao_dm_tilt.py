"""
Diagnostic script: Side-by-side comparison of OOPAO vs SAOS DM forward models.
Goal: Find exactly WHERE the two implementations diverge.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'OOPAO')))

from OOPAO.Telescope import Telescope as OOPAO_Tel
from OOPAO.Source import Source as OOPAO_Src
from OOPAO.DeformableMirror import DeformableMirror as OOPAO_DM
from OOPAO.calibration.get_modal_basis import get_projector as oopao_get_projector

# ═══════════════════════════════════════════
# SHARED PARAMETERS
# ═══════════════════════════════════════════
D           = 4.2
resolution  = 216
nSubap      = 36
mechCoupling= 0.60
validActThreshpercentage = 0.7533

# ═══════════════════════════════════════════
# 1. BUILD OOPAO DM
# ═══════════════════════════════════════════
print("="*60)
print("OOPAO SETUP")
print("="*60)
tel_o = OOPAO_Tel(resolution=resolution, diameter=D, centralObstruction=0.0)
ngs_o = OOPAO_Src(optBand='V', magnitude=10, coordinates=[0,0])
ngs_o * tel_o

dm_o = OOPAO_DM(telescope=tel_o, nSubap=nSubap, mechCoupling=mechCoupling)

print(f"OOPAO nValidAct: {dm_o.nValidAct}")
print(f"OOPAO nAct:      {dm_o.nAct}")
print(f"OOPAO modes shape: {dm_o.modes.shape}")
print(f"OOPAO pitch:     {dm_o.pitch}")
print(f"OOPAO D:         {dm_o.D}")
print(f"OOPAO resolution:{dm_o.resolution}")

# Build OOPAO basis (pupil-masked modes, exactly as their calibration code does)
M2C_o = np.eye(dm_o.nValidAct)
dm_o.coefs = M2C_o
tel_o * dm_o
basis_oopao = np.reshape(tel_o.OPD, [tel_o.resolution**2, M2C_o.shape[1]])
proj_oopao  = oopao_get_projector(basis_oopao)

print(f"OOPAO basis shape:     {basis_oopao.shape}")
print(f"OOPAO projector shape: {proj_oopao.shape}")

# ═══════════════════════════════════════════
# 2. BUILD SAOS DM (mimicking SAOS parameters)
# ═══════════════════════════════════════════
print("\n" + "="*60)
print("SAOS-STYLE COMPUTATION (standalone)")
print("="*60)

nActs = nSubap + 1
pitch_saos = D / (nActs - 1)
epsilon_saos = np.sqrt(-np.log(mechCoupling)) / pitch_saos

# Actuator coordinates (same as SAOS generate_cartesian_dm)
x_act = np.linspace(-D/2, D/2, nActs)
X_act, Y_act = np.meshgrid(x_act, x_act)
coords_all = np.column_stack([X_act.ravel(), Y_act.ravel()])

# Valid actuator mask (with threshold)
r_act = np.sqrt(coords_all[:,0]**2 + coords_all[:,1]**2)
valid_outer = D/2 + validActThreshpercentage * pitch_saos
validAct = r_act <= valid_outer
nValidAct_saos = np.sum(validAct)
coords_valid = coords_all[validAct]

print(f"SAOS nValidAct: {nValidAct_saos}")
print(f"SAOS nActs:     {nActs}")
print(f"SAOS pitch:     {pitch_saos}")
print(f"SAOS epsilon:   {epsilon_saos}")

# High-res grid (same as SAOS)
x_hr = np.linspace(-D/2, D/2, resolution)
X_hr, Y_hr = np.meshgrid(x_hr, x_hr)
hr_coords = np.column_stack([X_hr.ravel(), Y_hr.ravel()])

# Compute phi_eval (SAOS style via cdist)
from scipy.spatial.distance import cdist
D_eval = cdist(hr_coords, coords_valid)
phi_eval_saos = np.exp(-(epsilon_saos * D_eval)**2)

print(f"SAOS phi_eval shape: {phi_eval_saos.shape}")

# ═══════════════════════════════════════════
# 3. COMPARE IF MATRICES
# ═══════════════════════════════════════════
print("\n" + "="*60)
print("IF MATRIX COMPARISON")
print("="*60)

# OOPAO modes are the RAW modes (not pupil masked)
modes_oopao = dm_o.modes  # shape [res**2, nValidAct]

# Are the OOPAO valid actuators the same as SAOS?
print(f"OOPAO nValidAct: {dm_o.nValidAct}, SAOS nValidAct: {nValidAct_saos}")

if dm_o.nValidAct == nValidAct_saos:
    # Compare the IF matrices directly
    diff = np.abs(modes_oopao - phi_eval_saos)
    print(f"Max |modes_oopao - phi_eval_saos|: {diff.max():.2e}")
    print(f"Mean |modes_oopao - phi_eval_saos|: {diff.mean():.2e}")
    
    # Compare single actuator profiles
    mid_act = nValidAct_saos // 2
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    if_oopao_2d = modes_oopao[:, mid_act].reshape(resolution, resolution)
    if_saos_2d  = phi_eval_saos[:, mid_act].reshape(resolution, resolution)
    
    im1 = axs[0].imshow(if_oopao_2d, origin='lower', cmap='viridis')
    axs[0].set_title(f'OOPAO IF #{mid_act}')
    plt.colorbar(im1, ax=axs[0])
    
    im2 = axs[1].imshow(if_saos_2d, origin='lower', cmap='viridis')
    axs[1].set_title(f'SAOS IF #{mid_act}')
    plt.colorbar(im2, ax=axs[1])
    
    im3 = axs[2].imshow((if_oopao_2d - if_saos_2d)*1e6, origin='lower', cmap='seismic')
    axs[2].set_title(f'Difference ×1e6')
    plt.colorbar(im3, ax=axs[2])
    
    plt.suptitle('IF Matrix Comparison: Single Actuator')
    plt.tight_layout()
    plt.show()
else:
    print("WARNING: Different number of valid actuators! Cannot compare directly.")
    print(f"  OOPAO valid acts: {dm_o.nValidAct}")
    print(f"  SAOS  valid acts: {nValidAct_saos}")
    
    # Let's understand why
    print(f"\n  OOPAO valid threshold: D/2 + 0.7533*pitch = {dm_o.D/2 + 0.7533*dm_o.pitch:.4f}")
    print(f"  SAOS  valid threshold: D/2 + {validActThreshpercentage}*pitch = {D/2 + validActThreshpercentage*pitch_saos:.4f}")
    print(f"\n  OOPAO pitch: {dm_o.pitch:.6f}")
    print(f"  SAOS  pitch: {pitch_saos:.6f}")
    print(f"\n  OOPAO D: {dm_o.D}")
    print(f"  SAOS  D: {D}")

# ═══════════════════════════════════════════
# 4. COMPARE PROJECTORS (if same nValidAct)
# ═══════════════════════════════════════════
print("\n" + "="*60)
print("PROJECTOR COMPARISON")
print("="*60)

# OOPAO basis = modes * pupil
# SAOS projector = pinv(phi_eval * metapupil)

# Let's build the SAOS projector the same way
pupil_circle = (X_hr**2 + Y_hr**2) < ((resolution+1)/2 * D/resolution)**2
pupil_1d = pupil_circle.ravel().astype(float)

phi_masked_saos = phi_eval_saos * pupil_1d[:, None]
proj_saos = np.linalg.pinv(phi_masked_saos)

print(f"OOPAO projector shape: {proj_oopao.shape}")
print(f"SAOS  projector shape: {proj_saos.shape}")

# ═══════════════════════════════════════════
# 5. APPLY TILT AND COMPARE FITTING
# ═══════════════════════════════════════════
print("\n" + "="*60)
print("TILT FITTING COMPARISON")
print("="*60)

# Generate tilt target
tilt_amp = 1e-6
ideal_tilt = (X_hr / (D/2)) * tilt_amp
tel_pupil = tel_o.pupil
ideal_tilt_masked = ideal_tilt * tel_pupil

target_1d = ideal_tilt_masked.ravel()

# OOPAO fitting
coefs_oopao = proj_oopao @ target_1d
opd_oopao_raw = (modes_oopao @ coefs_oopao).reshape(resolution, resolution)
opd_oopao = opd_oopao_raw * tel_pupil

# SAOS fitting  
coefs_saos = proj_saos @ target_1d
opd_saos_raw = (phi_eval_saos @ coefs_saos).reshape(resolution, resolution)
opd_saos = opd_saos_raw * tel_pupil

# Residuals
res_oopao = (ideal_tilt_masked - opd_oopao) * tel_pupil
res_saos  = (ideal_tilt_masked - opd_saos) * tel_pupil

print(f"OOPAO coefs range: [{coefs_oopao.min():.4e}, {coefs_oopao.max():.4e}]")
print(f"SAOS  coefs range: [{coefs_saos.min():.4e}, {coefs_saos.max():.4e}]")
print(f"OOPAO residual RMS: {np.std(res_oopao[tel_pupil>0])*1e9:.2f} nm")
print(f"SAOS  residual RMS: {np.std(res_saos[tel_pupil>0])*1e9:.2f} nm")
print(f"OOPAO OPD PtV:      {(opd_oopao[tel_pupil>0].max()-opd_oopao[tel_pupil>0].min())*1e9:.2f} nm")
print(f"SAOS  OPD PtV:      {(opd_saos[tel_pupil>0].max()-opd_saos[tel_pupil>0].min())*1e9:.2f} nm")

fig, axs = plt.subplots(2, 4, figsize=(20, 10))

vmax = np.max(np.abs(ideal_tilt_masked))*1e9

axs[0,0].imshow(ideal_tilt_masked*1e9, origin='lower', cmap='seismic', vmin=-vmax, vmax=vmax)
axs[0,0].set_title('Target Tilt [nm]')

axs[0,1].imshow(opd_oopao*1e9, origin='lower', cmap='seismic', vmin=-vmax, vmax=vmax)
axs[0,1].set_title('OOPAO DM OPD [nm]')

axs[0,2].imshow(res_oopao*1e9, origin='lower', cmap='seismic')
axs[0,2].set_title(f'OOPAO Residual [nm]\nRMS={np.std(res_oopao[tel_pupil>0])*1e9:.1f} nm')

axs[0,3].imshow(opd_oopao_raw*1e9, origin='lower', cmap='seismic')
axs[0,3].set_title('OOPAO raw OPD (no pupil)')

axs[1,0].imshow(ideal_tilt_masked*1e9, origin='lower', cmap='seismic', vmin=-vmax, vmax=vmax)
axs[1,0].set_title('Target Tilt [nm]')

axs[1,1].imshow(opd_saos*1e9, origin='lower', cmap='seismic', vmin=-vmax, vmax=vmax)
axs[1,1].set_title('SAOS DM OPD [nm]')

axs[1,2].imshow(res_saos*1e9, origin='lower', cmap='seismic')
axs[1,2].set_title(f'SAOS Residual [nm]\nRMS={np.std(res_saos[tel_pupil>0])*1e9:.1f} nm')

axs[1,3].imshow(opd_saos_raw*1e9, origin='lower', cmap='seismic')
axs[1,3].set_title('SAOS raw OPD (no pupil)')

plt.suptitle('OOPAO (top) vs SAOS (bottom) - Tilt Fitting', fontsize=14)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════
# 6. COMPARE ACTUATOR COORDINATES
# ═══════════════════════════════════════════
print("\n" + "="*60)
print("ACTUATOR COORDINATES")
print("="*60)
print(f"OOPAO coords shape: {dm_o.coordinates.shape}")
print(f"SAOS  coords shape: {coords_valid.shape}")
if dm_o.nValidAct == nValidAct_saos:
    coord_diff = np.abs(dm_o.coordinates - coords_valid)
    print(f"Max coordinate difference: {coord_diff.max():.2e} m")
