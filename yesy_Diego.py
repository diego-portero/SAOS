# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:01:52 2024

@author: cheritier


Tutorial Description — Shack-Hartmann Wave-Front Sensing in OOPAO

This tutorial provides a focused walkthrough of Shack-Hartmann (SH) wave-front sensing in OOPAO, 
illustrating how to configure, sample, visualize, and calibrate a SH sensor within a full optical chain.

Starting from a telescope, guide star, and optional atmosphere, the script introduces the SH WFS as a modular optical element, showing how it receives the propagated NGS signal via the * operator.
It compares several SH sampling regimes:
    - Shannon/2
    - Shannon
    - custom pixel scales
    
Users can inspect raw camera frames, slope maps, valid-subaperture masks, and cubes of individual subaperture spots.

The tutorial highlights practical SH utilities:
    - enabling/disabling photon-noise
    - switching between diffractive and geometric (gradient-only) sensing
    - applying Gaussian centroid-weighting maps to optimize slope estimation. 

A KL modal basis is generated and used to build a modal interaction matrix with the SH sensor.

The calibrated interaction matrix is then used to set up a closed-loop environment, where the SH slopes drive the DM via a modal reconstructor.

"""

# -*- coding: utf-8 -*-
"""
Simulation Script: yesy_Diego.py
Author: Adapted for SAOS 2026
"""

from datetime import datetime
import time
from pathlib import Path

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
from SAOS.CorrelatingShackHartmann import CorrelatingShackHartmann
from SAOS.LightPath import LightPath
from SAOS.InteractionMatrixHandler import InteractionMatrixHandler
from SAOS.ScienceCam import ScienceCam
from SAOS.Sharepoint import Sharepoint
from SAOS.Savepoint import Savepoint
from SAOS.ReceptionPoint import ReceptionPoint
from SAOS.IP_WFS import IP_WFS
from SAOS.Controller import Controller

# Logger:

test_logger = LoggingHelper(logging.INFO)

# Simulation settings:

nIterations = 500

generate_new_atm = False
measure_new_IM = False
load_modal_basis = True

g = 0.2
decay = 0.9999
decimate = 1 # decimation for the Sharepoint (GUI)
im_stroke = 0.75e-6 # in meters
loop_status = 1 # If 1, close loop. Open otherwise.

home = str(Path.home())

# # Loading files:
# load_filename_atm = '/home/dportero/Tesis/SAOS/simulations/simulations/phase_screens/20260220_1144.h5'
# load_filename_modalBasis = '/home/dportero/Tesis/SAOS/simulations/simulations/modal_basis/20260220_1657.h5'
# load_filename_IM = '/home/dportero/Tesis/SAOS/simulations/simulations/interaction_matrix/20260220_1657.h5'

# Saving files
date = datetime.datetime.now().strftime("%Y%m%d_%H%M")

save_filename_atm = '/home/dportero/Tesis/SAOS/simulations_5/phase_screens/' + date
save_filename_modalBasis = '/home/dportero/Tesis/SAOS/simulations_5/modal_basis/' + date
save_filename_IM = '/home/dportero/Tesis/SAOS/simulations_5/interaction_matrix/' + date

# Define data sharepoint

# sharepoint = Sharepoint(test_logger.logger, port=5559, atm=1, atm_per_dir=1, dm=1, dm_per_dir=1, slopes=1, wfs=1, wfs_frame=1, sci=1, sci_frame=1)

# Define ReceptionPoint

# receptionpoint = ReceptionPoint(ip='localhost', port='7001', logger=test_logger.logger)

# Define the savingpoint
save_filename_data = f'/home/dportero/Tesis/SAOS/simulations_5/savepoints/saos_savepoint_{date}.h5'

savepoint = Savepoint(
    file_path=save_filename_data,
    atm=1,
    atm_per_dir=1,
    dm=1,
    dm_per_dir=1,
    slopes=1,
    wfs=1,
    wfs_frame=1,
    sci=1,
    sci_frame=1,
    logger=test_logger.logger
)

# Define EST
t0 = time.time()

diameter = 4.2 # in [m]
obs_diameter = 1.3 # in [m]
sampling_time = 1/1000 # in [s]
n_subaperture_red = 20
resolution = n_subaperture_red * 6 # resolution of the phase screen in [px]
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

# Atmosphere:

atm = Atmosphere(r0 = 0.07,
                 L0= 25,
                 fractionalR0=[0.6, 0.2, 0.15, 0.02, 0.02, 0.01],
                 altitude=[0, 2000, 5000, 10000, 15000, 20000],
                 windDirection=[0, 20, 10, 45, 90, 60],
                 windSpeed=[10, 19, 25, 15, 25, 19],
                 telescope=est_tel,
                 logger=test_logger.logger)


atm.initializeAtmosphere()
atm.save(save_filename_atm)

# Sources:
sun_red = ExtendedSource(optBand = 'R',
                       coordinates=[0, 0],
                       nSubDirs=3,
                       fov=9.269,
                       subDir_margin=4.0,
                       patch_padding=5.0,
                       logger=test_logger.logger)

ngs = Source(magnitude = 5,
             optBand = 'R4',
             coordinates=[0,0],
             logger=test_logger.logger)

# Deformable mirrors:

asm = DeformableMirror(telescope=est_tel,
                        nSubap=n_subaperture_red,
                        altitude=0,
                        logger=test_logger.logger) # ASM

dms = [asm]  


# Deformable mirrors:
nSubap_WFS = n_subaperture_red
OPD = atm.layer_1.screen.scrn  # must be in meters

red_wfs = IP_WFS(nSubap=int(np.sqrt(nSubap_WFS)**2), 
                 telescope=est_tel,
                 src=sun_red,
                 lightRatio=0.9,
                 plate_scale=0.403,
                 fieldOfView=9.269,
                 guardPx=2,
                 OPD=OPD
                 )

wavefront = red_wfs.wfs_measure(
    OPD,
    sun_red
)
# red_wfs.plot_debug()

# import sys
# sys.exit()

# Science camera

science_cam = ScienceCam(fieldOfView=9.269, 
                         plate_scale = 0.0167,
                         samplingTime=est_tel.samplingTime,
                         lightRatio=0.15,
                         telescope=est_tel,
                         decimation=decimate,
                         noiseFlag=True,
                         logger=test_logger.logger)

# Build the Light Path

scao_light_path_list = []

scao_light_path_list.append(LightPath(test_logger.logger))
scao_light_path_list[0].initialize_path(src=sun_red, atm=atm, tel=est_tel, dm=asm, wfs=red_wfs, ncpa=None, sci=science_cam)

scao_light_path_list.append(LightPath(test_logger.logger))
scao_light_path_list[1].initialize_path(src=ngs, atm=atm, tel=est_tel, dm=asm, ncpa=None, sci=science_cam)

test_logger.logger.info(f'The Modules initialization took {time.time()-t0} [s]')

# Now, we need to measure the IM of the Light Path
t0 = time.time()

im_handler = InteractionMatrixHandler(test_logger.logger)
im_handler.initialize_im_class(scao_light_path_list)


im_handler.measure(modal_basis='zernike', stroke=im_stroke, nModes=None)
im_handler.save_IM(save_filename_IM)


test_logger.logger.info(f'The IM creation took {time.time()-t0} [s]')



# Controller class
controller_kwargs = {'rcond':0.025, 
                    'beta':1e-4,
                    'gain':[0.2],
                    'decay':[0.99],
                    'ki':[0.0]}

controller = Controller(telescope=est_tel,
                        interactionMatrix=im_handler,
                        reconstructionMethod='tikhonov',
                        controllerType='leaky',
                        logger=test_logger.logger,
                        **controller_kwargs)

cmd_modal = 0

test_logger.logger.info('Beginning simulation')

# SCAO loop
for i in range(nIterations):
    
    test_logger.logger.info(f'Iteration {i+1}/{nIterations} ')    
    # Update the atmosphere
    atm.update()
    # Propagate the light
    scao_light_path_list[0].propagate(True)
    scao_light_path_list[1].propagate(True)
    # Compute command
    cmd = controller.computeControlAction(scao_light_path_list)
    # Update the DM shape
    for j in range(len(dms)):
        dms[j].updateDMShape(cmd[j])
    # Share data with the GUI
    #sharepoint.shareData(scao_light_path_list, i, [atm], [asm])              
 
    # Save Data
    savepoint.save([atm], i)
    savepoint.save([asm], i)
    savepoint.save(scao_light_path_list, i)

    # Ask the GUI if the user wants to open/close the loop 

    # new_loop_status = receptionpoint.sendRequest('loop_cmd')

    # if new_loop_status is not None:
    #     loop_status = new_loop_status

# Force destructor call for the qeue of logs

test_logger = None