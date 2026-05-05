import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import datetime
import logging
import numpy as np
import gc

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from joblib import Parallel, delayed

from SAOS.LoggingHelper import LoggingHelper
from SAOS.Atmosphere import Atmosphere
from SAOS.Telescope import Telescope
from SAOS.ExtendedSource import ExtendedSource
from SAOS.Source import Source
from SAOS.ShackHartmann import ShackHartmann
from SAOS.CorrelatingShackHartmann import CorrelatingShackHartmann
from SAOS.ScienceCam import ScienceCam
from SAOS.DeformableMirror import DeformableMirror
from SAOS.LightPath import LightPath
from SAOS.InteractionMatrixHandler import InteractionMatrixHandler
from SAOS.Controller import Controller
from SAOS.Sharepoint import Sharepoint
from SAOS.Savepoint import Savepoint
# Logger:

test_logger = LoggingHelper(logging.INFO)

# Simulation settings:

nIterations = 3000

scienceFs = 56. # Hz

generate_new_atm = False
measure_new_IM = True
load_modal_basis = False

nModes = None # [nModesASM, nModesM7]
im_stroke = [5e-7] # in meters

# Loading files:
load_filename_atm = os.path.join(os.path.expanduser("~"), 'simulations/phase_screens/20260504_1038.h5')
load_filename_modalBasis = os.path.join(os.path.expanduser("~"), 'simulations/modal_basis/20260504_1032.h5')
load_filename_IM = os.path.join(os.path.expanduser("~"), 'simulations/interaction_matrix/20260223_1353.h5')

# Saving files
date = datetime.datetime.now().strftime("%Y%m%d_%H%M")

save_filename_atm = os.path.join(os.path.expanduser("~"), 'simulations/phase_screens/' + date + '.h5')
save_filename_modalBasis = os.path.join(os.path.expanduser("~"), 'simulations/modal_basis/' + date + '.h5')
save_filename_IM = os.path.join(os.path.expanduser("~"), 'simulations/interaction_matrix/' + date + '.h5')

## Define data sharepoint

sharepoint = Sharepoint(test_logger.logger, port=5573, atm=1, atm_per_dir=0, dm=1, dm_per_dir=0, slopes=1, wfs=1, wfs_frame=1, sci=1, sci_frame=1)

## Define the savingpoint
savepoint = Savepoint(file_path='', atm=1, atm_per_dir=1, dm=1, dm_per_dir=1, slopes=1, wfs=1, wfs_frame=1, sci=1, sci_frame=1, only_metrics=1, logger=test_logger.logger)

## Define EST
t0 = time.time()

diameter = 4.149 # in [m]
obs_diameter = 1.3 # in [m]
sampling_time = 1/2000 # in [s]
n_subaperture_red = 36
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

## Atmosphere:

atm = Atmosphere(r0 = 0.08,
                 L0= 25,
                 fractionalR0=[0.53, 0.37, 0.05, 0.03, 0.02],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[45, 90, 180, 67, 2],
                 windSpeed=[2.8, 5.3, 9.6, 18.4, 17.8],
                 telescope=est_tel,
                 zenith = 0,
                 logger=test_logger.logger)

if generate_new_atm:
    atm.initializeAtmosphere()
    atm.save(save_filename_atm)
else:
    atm.load(load_filename_atm)

## Sources:
ngs_red = Source(magnitude = 5,
             optBand = 'R4',
             coordinates=[0,0],
             logger=test_logger.logger)

sun_red = ExtendedSource(optBand = 'R',
                     coordinates=[0, 0],
                     nSubDirs=3,
                     fov=9.269,
                     subDir_margin=4.0,
                     patch_padding=5.0,
                     logger=test_logger.logger)             

## Deformable mirrors:

asm_params = {'dynamicModel': '', 'validThreshPercentage': 0.5}
# asm_params = {'dynamicModel': ''}

asm = DeformableMirror(telescope=est_tel,
                        nActs=n_subaperture_red+1,
                        altitude=0,
                        typeDM='radial',
                        logger=test_logger.logger,
                        **asm_params) # ASM

dms = [asm]

## Wavefront Sensor

red_wfs = CorrelatingShackHartmann(telescope=est_tel,
                                    src=sun_red,
                                    lightRatio=0.9,
                                    nSubap=n_subaperture_red,
                                    plate_scale=0.403,
                                    fieldOfView=9.269,
                                    guardPx=2,
                                    fft_fieldOfView_oversampling=0.5,
                                    use_brightest=9,
                                    unit_in_rad=False,
                                    logger=test_logger.logger)

red_wfs_p = ShackHartmann(telescope=est_tel,
                        src=ngs_red,
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

scao_light_path_list = []

# Create red branch
scao_light_path_list.append(LightPath(test_logger.logger))
scao_light_path_list[-1].initialize_path(src=ngs_red, atm=atm, tel=est_tel, dm=dms[0], wfs=red_wfs_p, vibration=None, sci=red_scicam, delay=1)

lightPathTasks = []
for i in range(len(scao_light_path_list)):
    lightPathTasks.append(delayed(scao_light_path_list[i].propagate)(True))

test_logger.logger.info(f'The Modules initialization took {time.time()-t0} [s]')

## Interaction Matrix
t0 = time.time()

im_handler = InteractionMatrixHandler(test_logger.logger)
im_handler.initialize_im_class(scao_light_path_list)

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

test_logger.logger.info('Beginning simulation')

# SCAO loop
for i in range(nIterations):
    est_tel.logger.info(f'Iteration {i+1}')
    # Update the atmosphere
    atm.update()
    # Propagate the light
    Parallel(n_jobs=1, prefer="threads")(lightPathTasks)
    # Compute command
    cmd = controller.computeControlAction(scao_light_path_list)
    # Update the DM shape
    for j in range(len(dms)):
        dms[j].updateDMShape(cmd[j])
    # Share data with the GUI
    sharepoint.shareData(scao_light_path_list, i, [atm], dms)              
 
    test_logger.logger.info(f'PSF peak: {scao_light_path_list[0].sci_frame.max()}')
    # Save Data
    savepoint.save([atm], i)
    savepoint.save(dms, i)
    savepoint.save(scao_light_path_list, i)   

    if (i + 1) % 100 == 0:
        if HAS_TORCH:
            torch.cuda.empty_cache()
        gc.collect()

test_logger.logger.info('Simulation ended')
    
# Force destructor
try:
    del est_tel, atm, asm, red_wfs, controller, im_handler, scao_light_path_list, savepoint
except NameError:
    pass
test_logger = None

if HAS_TORCH:
    torch.cuda.empty_cache()
gc.collect()