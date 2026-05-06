from datetime import datetime
import time
import os
import gc

import matplotlib.pyplot as plt
import numpy as np

from joblib import Parallel, delayed

import logging

from SAOS.LoggingHelper import LoggingHelper
from SAOS.Source import Source
from SAOS.ExtendedSource import ExtendedSource
from SAOS.Telescope import Telescope
from SAOS.Atmosphere import Atmosphere
from SAOS.DeformableMirror import DeformableMirror
from SAOS.Vibration import Vibration
from SAOS.CorrelatingShackHartmann import CorrelatingShackHartmann
from SAOS.LightPath import LightPath
from SAOS.InteractionMatrixHandler import InteractionMatrixHandler
from SAOS.Controller import Controller
from SAOS.ScienceCam import ScienceCam
from SAOS.Sharepoint import Sharepoint
from SAOS.Savepoint import Savepoint

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Logger:

test_logger = LoggingHelper(logging.INFO)

# Simulation settings:

nIterations = 2000

scienceFs = 56. # Hz

generate_new_atm = False
measure_new_IM = False
load_modal_basis = True

nModes = None # [nModesASM, nModesM7]
im_stroke = [5e-7, 12e-7, 6.7e-7] # in meters

# Loading files:
load_filename_atm = os.path.join(os.path.expanduser("~"), 'simulations/phase_screens/ps_mcaoExample_r12cm.h5')
load_filename_modalBasis = os.path.join(os.path.expanduser("~"), 'simulations/modal_basis/modalBasis_mcaoExample.h5')
load_filename_IM = os.path.join(os.path.expanduser("~"), 'simulations/interaction_matrix/IM_mcaoExample.h5')

# Saving files
date = datetime.now().strftime("%Y%m%d_%H%M")

save_filename_atm = os.path.join(os.path.expanduser("~"), 'simulations/phase_screens/' + date + '.h5')
save_filename_modalBasis = os.path.join(os.path.expanduser("~"), 'simulations/modal_basis/' + date + '.h5')
save_filename_IM = os.path.join(os.path.expanduser("~"), 'simulations/interaction_matrix/' + date + '.h5')

## Define data sharepoint

sharepoint = Sharepoint(test_logger.logger, port=5574, atm=1, atm_per_dir=0, dm=1, dm_per_dir=0, slopes=1, wfs=1, wfs_frame=1, sci=1, sci_frame=1)

## Define the savingpoint
savepoint = Savepoint(file_path='', atm=1, atm_per_dir=1, dm=1, dm_per_dir=1, slopes=1, wfs=1, wfs_frame=1, sci=1, sci_frame=1, only_metrics=1, logger=test_logger.logger)

## Define EST
t0 = time.time()

diameter = 4.149 # in [m]
obs_diameter = 1.3 # in [m]
sampling_time = 1/2000 # in [s]
n_subaperture_red = 36
resolution = n_subaperture_red * 4 # resolution of the phase screen in [px]
pixel_size = diameter / resolution
tel_fov = 60 # in [arcsec]

est_tel = Telescope(diameter = diameter,
                    resolution = resolution,
                    centralObstruction= obs_diameter / diameter,
                    samplingTime=sampling_time,
                    fov=tel_fov,
                    logger=test_logger.logger)

spider_angle = [0, 90, 180, 270] # in [º]
spider_thickness = 0.060 # in [m]

# est_tel.apply_spiders(spider_angle, spider_thickness)

## Atmosphere:

atm = Atmosphere(r0=0.08,
                 L0=25,
                 fractionalR0=[0.53, 0.37, 0.05, 0.03, 0.02],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[312, 47, 198, 123, 276],
                 windSpeed=[2.8, 5.3, 9.6, 18.4, 17.8],
                 telescope=est_tel,
                 zenith=0,
                 logger=test_logger.logger)

if generate_new_atm:
    atm.initializeAtmosphere()
    atm.save(save_filename_atm)
else:
    atm.load(load_filename_atm)

## Sources:
# On axis
# 6 Extended Sources circularly distributed at a radious of 15" and rotated 45º
coord_list = [[0, 0]] + [[15, i * 360 / 6 + 45] for i in range(6)]

sun_list = []

for i in range(len(coord_list)):
    sun_list.append(ExtendedSource(optBand='R',
                                   coordinates=coord_list[i],
                                   nSubDirs=3,
                                   fov=9.269,
                                   subDir_margin=4.0,
                                   patch_padding=5.0,
                                   logger=test_logger.logger))
    

coord_list_field = [[0, 0]] + [[3.75, i * 360 / 4] for i in range(4)] + [[7.5, i * 360 / 4] for i in range(4)] + [[15, i * 360 / 4] for i in range(4)] + [[20, i * 360 / 4] for i in range(4)] + [[25, i * 360 / 4] for i in range(4)]

sci_list = []

for i in range(len(coord_list_field)):
    sci_list.append(Source(magnitude = 5,
                           optBand = 'R4',
                           coordinates=coord_list_field[i],
                           logger=test_logger.logger))

## Deformable mirrors:

asm_params = {'dynamicModel': '', 'validActThreshpercentage': 0.7533}

asm = DeformableMirror(telescope=est_tel,
                        nActs=n_subaperture_red+1,
                        altitude=0,
                        typeDM='cartesian',
                        logger=test_logger.logger,
                        **asm_params) # ASM

m3_params = {'dynamicModel': '', 'validActThreshpercentage': 0.7533}

m3 = DeformableMirror(telescope=est_tel,
                        nActs=25,
                        altitude=20000,
                        typeDM='cartesian',
                        logger=test_logger.logger,
                        **m3_params) # M3

m6_params = {'dynamicModel': '', 'validActThreshpercentage': 0.7533}

m6 = DeformableMirror(telescope=est_tel,
                        nActs=25,
                        altitude=5000,
                        typeDM='cartesian',
                        logger=test_logger.logger,
                        **m6_params) # M6

dms = [asm, m3, m6]

## Wavefront Sensor

red_wfs = CorrelatingShackHartmann(telescope=est_tel,
                                    src=sun_list[0],
                                    lightRatio=0.9,
                                    nSubap=n_subaperture_red,
                                    plate_scale=0.403,
                                    fieldOfView=9.269,
                                    guardPx=2,
                                    fft_fieldOfView_oversampling=0.5,
                                    use_brightest=9,
                                    unit_in_rad=False,
                                    logger=test_logger.logger)

## Science camera

red_scicam = ScienceCam(fieldOfView=9.269, 
                         plate_scale = 0.0167,
                         samplingTime=est_tel.samplingTime,
                         telescope=est_tel,
                         integrationTime=1./scienceFs,
                         noiseFlag=False,
                         logger=test_logger.logger)

## Build the Light Path

light_path_list = []

# Create red branch: sensing
for i in range(len(sun_list)):
    light_path_list.append(LightPath(test_logger.logger))
    light_path_list[-1].initialize_path(src=sun_list[i], atm=atm, tel=est_tel, dm=dms, wfs=red_wfs, vibration=None, sci=None, delay=0)

for i in range(len(sci_list)):
    light_path_list.append(LightPath(test_logger.logger))
    light_path_list[-1].initialize_path(src=sci_list[i], atm=atm, tel=est_tel, dm=dms, wfs=None, vibration=None, sci=red_scicam, delay=0)

lightPathTasks = []
for i in range(len(light_path_list)):
    lightPathTasks.append(delayed(light_path_list[i].propagate)(True))

test_logger.logger.info(f'The Modules initialization took {time.time()-t0} [s]')

## Interaction Matrix
t0 = time.time()

im_handler = InteractionMatrixHandler(test_logger.logger)
im_handler.initialize_im_class(light_path_list)

if load_modal_basis:
    im_handler.load_modalBasis(load_filename_modalBasis)
if measure_new_IM:
    im_handler.measure(modal_basis='dh', stroke=im_stroke, nModes=nModes)
    im_handler.save_IM(save_filename_IM)

    if not load_modal_basis:
        im_handler.save_modalBasis(save_filename_modalBasis)
else:
    im_handler.load_IM(load_filename_IM)

test_logger.logger.info(f'The IM creation took {time.time()-t0} [s]')

# Define explicit control mask: 3 DMs x LPs
# We want all 3 DMs to use the 7 WFSs (LPs 0-6), and ignore the science camera (LP 7-end)
control_mask = [[True]*len(sun_list) + [False]*len(sci_list) for _ in range(3)]

# Controller class
controller_kwargs = {'rcond':0.025, 
                    'beta':1e-4,
                    'gain':[0.4, 0.15, 0.15],
                    'decay':[0.9999, 0.99, 0.99],
                    'ki':[0.0, 0.0, 0.0],
                    'control_mask': control_mask}

controller = Controller(telescope=est_tel,
                        interactionMatrix=im_handler,
                        reconstructionMethod='tikhonov',
                        controllerType='leaky',
                        logger=test_logger.logger,
                        **controller_kwargs)

test_logger.logger.info('Beginning simulation')

# MCAO loop
for i in range(nIterations):
    est_tel.logger.info(f'Iteration {i+1}')
    # Update the atmosphere
    atm.update()
    # Propagate the light
    Parallel(n_jobs=2, prefer="threads")(lightPathTasks)
    # Compute command
    cmd = controller.computeControlAction(light_path_list)
    # Update the DM shape
    for j in range(len(dms)):
        dms[j].updateDMShape(cmd[j])
    # Share data with the GUI
    # sharepoint.shareData(light_path_list, i, [atm], dms)              
    
    # Log peak PSFs from science cameras
    # LPs 0-6 are sensing (sun_list), LPs 7-end are science (sci_list)
    psf_0 = np.max(light_path_list[7].sci_frame) if light_path_list[7].sci_frame is not None else 0
    psf_1 = np.max(light_path_list[8].sci_frame) if light_path_list[8].sci_frame is not None else 0
    psf_2 = np.max(light_path_list[12].sci_frame) if light_path_list[12].sci_frame is not None else 0
    psf_3 = np.max(light_path_list[16].sci_frame) if light_path_list[16].sci_frame is not None else 0
    psf_4 = np.max(light_path_list[20].sci_frame) if light_path_list[20].sci_frame is not None else 0
    psf_5 = np.max(light_path_list[24].sci_frame) if light_path_list[24].sci_frame is not None else 0

    test_logger.logger.info(f'Peak PSFs -> 0": {psf_0:.4f} | 3.75": {psf_1:.4f} | 7.5": {psf_2:.4f} | 15": {psf_3:.4f} | 20": {psf_4:.4f} | 25": {psf_5:.4f}')
    
    # Save Data
    savepoint.save([atm], i)
    savepoint.save(dms, i)
    savepoint.save(light_path_list, i)

    if (i + 1) % 100 == 0:
        if HAS_TORCH:
            torch.cuda.empty_cache()
        gc.collect()

test_logger.logger.info('Simulation ended.')

# Force destructor call for the qeue of logs
try:
    del est_tel, atm, asm, m3, m6, red_wfs, controller, im_handler, light_path_list, savepoint
except NameError:
    pass
test_logger = None

if HAS_TORCH:
    torch.cuda.empty_cache()
gc.collect()