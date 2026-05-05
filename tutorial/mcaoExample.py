from datetime import datetime
import time
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from joblib import Parallel, delayed

import logging
import datetime

from SAOS.LoggingHelper import LoggingHelper
from SAOS.Source import Source
from SAOS.ExtendedSource import ExtendedSource
from SAOS.Telescope import Telescope
from SAOS.Atmosphere import Atmosphere
from SAOS.DeformableMirror import DeformableMirror
from SAOS.Vibration import Vibration
from SAOS.ShackHartmann import ShackHartmann
from SAOS.CorrelatingShackHartmann import CorrelatingShackHartmann
from SAOS.LightPath import LightPath
from SAOS.InteractionMatrixHandler import InteractionMatrixHandler
from SAOS.Controller import Controller
from SAOS.ScienceCam import ScienceCam
from SAOS.Sharepoint import Sharepoint
from SAOS.Savepoint import Savepoint

# Logger:

test_logger = LoggingHelper(logging.INFO)

# Simulation settings:

nIterations = 250

scienceFs = 56. # Hz

generate_new_atm = True
measure_new_IM = True
load_modal_basis = False

nModes = None # [nModesASM, nModesM7]
im_stroke = [5e-7, 1.5e-6, 1.5e-6] # in meters

# Loading files:
load_filename_atm = os.path.join(os.path.expanduser("~"), 'simulations/phase_screens/mcaoExample.h5')
load_filename_modalBasis = os.path.join(os.path.expanduser("~"), 'simulations/modal_basis/mcaoExample.h5')
load_filename_IM = os.path.join(os.path.expanduser("~"), 'simulations/interaction_matrix/mcaoExample.h5')

# Saving files
date = datetime.datetime.now().strftime("%Y%m%d_%H%M")

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

atm = Atmosphere(r0 = 0.40,
                 L0= 25,
                 fractionalR0=[0.29, 0.19, 0.20, 0.19, 0.14],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[0, 45, 225, 315, 135],
                 windSpeed=[8, 19, 20, 17, 23],
                 telescope=est_tel,
                 zenith = 60,
                 logger=test_logger.logger)

if generate_new_atm:
    atm.initializeAtmosphere()
    atm.save(save_filename_atm)
else:
    atm.load(load_filename_atm)

## Sources:
# On axis
# 6 NGS cicurlarly distributed at a radious of 15" and rotated 45º
coord_list = [[0, 0]] + [[15, i * 360 / 6 + 45] for i in range(6)]

ngs_list = []

for i in range(len(coord_list)):
    ngs_list.append(Source(magnitude = 5,
                           optBand = 'R4',
                           coordinates=coord_list[i],
                           logger=test_logger.logger))
    

ext_sci_ngs = Source(magnitude = 5,
                    optBand = 'R4',
                    coordinates=[25,30],
                    logger=test_logger.logger)

## Deformable mirrors:

asm_params = {'dynamicModel': '', 'validActThreshpercentage': 0.7533}

asm = DeformableMirror(telescope=est_tel,
                        nActs=n_subaperture_red+1,
                        altitude=0,
                        typeDM='radial',
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

red_wfs = ShackHartmann(telescope=est_tel,
                      src=ngs_list[0],
                      lightRatio=0.9,
                      nSubap=n_subaperture_red,
                      plate_scale=0.403,
                      fieldOfView=9.269,
                      guardPx=2,
                      fft_fieldOfView_oversampling=0.5,
                      use_brightest=50,
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

# Create red branch
for i in range(len(ngs_list)):
    if (i == 0) or (i == 2) or (i == 6): # Add science camera
        light_path_list.append(LightPath(test_logger.logger))
        light_path_list[-1].initialize_path(src=ngs_list[i], atm=atm, tel=est_tel, dm=dms, wfs=red_wfs, vibration=None, sci=red_scicam, delay=0)
    else:
        light_path_list.append(LightPath(test_logger.logger))
        light_path_list[-1].initialize_path(src=ngs_list[i], atm=atm, tel=est_tel, dm=dms, wfs=red_wfs, vibration=None, sci=None, delay=0)

light_path_list.append(LightPath(test_logger.logger))
light_path_list[-1].initialize_path(src=ext_sci_ngs, atm=atm, tel=est_tel, dm=dms, wfs=None, vibration=None, sci=red_scicam, delay=0)

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

# Define explicit control mask: 3 DMs x 8 LPs
# We want all 3 DMs to use the 7 WFSs (LPs 0-6), and ignore the science camera (LP 7)
control_mask = [[True]*7 + [False] for _ in range(3)]

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
    
    # Log peak PSFs from science cameras (LPs 0, 2, 6, and 7)
    psf_0 = np.max(light_path_list[0].sci_frame)
    psf_2 = np.max(light_path_list[2].sci_frame)
    psf_6 = np.max(light_path_list[6].sci_frame)
    psf_7 = np.max(light_path_list[7].sci_frame)
    test_logger.logger.info(f'Peak PSFs -> LP0: {psf_0:.4f} | LP2: {psf_2:.4f} | LP6: {psf_6:.4f} | ExtSci: {psf_7:.4f}')
    
    # Save Data
    # savepoint.save([atm], i)
    # savepoint.save(dms, i)
    # savepoint.save(light_path_list, i)

test_logger.logger.info('Simulation ended.')

# Force destructor call for the qeue of logs

test_logger = None