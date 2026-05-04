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
from SAOS.ExtendedSource import ExtendedSource
from SAOS.Telescope import Telescope
from SAOS.Atmosphere import Atmosphere
from SAOS.DeformableMirror import DeformableMirror
from SAOS.CorrelatingShackHartmann import CorrelatingShackHartmann
from SAOS.LightPath import LightPath
from SAOS.InteractionMatrixHandler import InteractionMatrixHandler
from SAOS.Controller import Controller
from SAOS.ScienceCam import ScienceCam
from SAOS.Savepoint import Savepoint
from SAOS.Source import Source

if __name__ == '__main__':
    base_dir = os.path.join(os.path.expanduser("~"), 'simulations')
        
    # Directories setup based on base directory
    phase_screens_dir = os.path.join(base_dir, 'phase_screens')
    modal_basis_dir = os.path.join(base_dir, 'modal_basis')
    im_dir = os.path.join(base_dir, 'interaction_matrix')
    mirror_models_dir = os.path.join(base_dir, 'MirrorModels')
    vibrations_dir = os.path.join(base_dir, 'VibrationsSource')
    results_dir = os.path.join(base_dir, 'results')   

    os.makedirs(phase_screens_dir, exist_ok=True)
    os.makedirs(modal_basis_dir, exist_ok=True)
    os.makedirs(im_dir, exist_ok=True)
    os.makedirs(mirror_models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # File paths specific to this combination
    atm_filename = os.path.join(phase_screens_dir, 'ps_scao_EST.h5')
    load_filename_modalBasis = os.path.join(modal_basis_dir, 'scaoEST.h5')
    load_filename_IM = os.path.join(im_dir, 'scaoEST.h5')
    
    date = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    res_base = f'res_scao_atm1_{date}.h5'
    save_filename_IM = load_filename_IM
    save_filename_modalBasis = load_filename_modalBasis

    test_logger = LoggingHelper(logging.INFO)
    test_logger.logger.info("=== STARTED SCAO SOLAR EXAMPLE ===")

    # Simulation settings:
    nIterations = 3000
    scienceFs = 56. # Hz

    measure_new_IM = True
    load_modal_basis = True

    nModes = [500]
    im_stroke = [1.5e-6]

    save_path_prefix = os.path.join(results_dir, res_base)
    savepoint = Savepoint(file_path=save_path_prefix, atm=1, atm_per_dir=1, dm=1, dm_per_dir=1, slopes=1, wfs=1, wfs_frame=1, sci=1, sci_frame=1, only_metrics=1, logger=test_logger.logger)

    t0 = time.time()
    diameter = 4.149
    obs_diameter = 1.3
    sampling_time = 1/2000
    n_subaperture_red = 36
    resolution = n_subaperture_red * 6
    pixel_size = diameter / resolution
    tel_fov = 60

    est_tel = Telescope(diameter=diameter, resolution=resolution, centralObstruction=obs_diameter/diameter, samplingTime=sampling_time, fov=tel_fov, logger=test_logger.logger)

    # Atmosphere (Case 1 modified to r0=0.08)
    atm_kwargs = {
        'r0': 0.08, 'L0': 25, 'zenith': 15,
        'fractionalR0': [0.53, 0.37, 0.05, 0.03, 0.02],
        'altitude': [100, 1500, 5000, 10000, 15000],
        'windSpeed': [0,0,0,0,0],#[2.8, 5.3, 9.6, 18.4, 17.8]
        'windDirection': [312, 47, 198, 123, 276]
    }
    atm = Atmosphere(telescope=est_tel, logger=test_logger.logger, **atm_kwargs)

    # Check if atm exists to generate or load
    if os.path.exists(atm_filename):
        test_logger.logger.info(f"Loading atmosphere from {atm_filename}")
        atm.load(atm_filename)
    else:
        test_logger.logger.info(f"Generating new atmosphere phase screen and saving to {atm_filename}")
        atm.initializeAtmosphere()
        atm.save(atm_filename)

    # Sources
    ngs_red = Source(magnitude=5, 
                     optBand='R4', 
                     coordinates=[0,0], 
                     logger=test_logger.logger)
    sun_red = ExtendedSource(optBand='R', 
                             coordinates=[0, 0], 
                             nSubDirs=3, 
                             fov=9.269, 
                             subDir_margin=4.0, 
                             patch_padding=5.0, 
                             logger=test_logger.logger)             

    # Deformable mirrors
    asm_params = {'dynamicModel': os.path.join(mirror_models_dir, 'asm_discrete_model.h5'), 'validActThreshpercentage': 0.7533}
    asm = DeformableMirror(telescope=est_tel, 
                            nActs=n_subaperture_red+1, 
                            altitude=0, 
                            typeDM='radial', 
                            logger=test_logger.logger, 
                            **asm_params)

    dms = [asm]

    # Vibrations
    red_vibrations = None

    # WFS
    red_wfs = CorrelatingShackHartmann(telescope=est_tel, src=sun_red, lightRatio=0.9, nSubap=n_subaperture_red, plate_scale=0.403, fieldOfView=9.269, guardPx=2, fft_fieldOfView_oversampling=0.5, use_brightest=9, unit_in_rad=False, logger=test_logger.logger)

    red_scicam = ScienceCam(fieldOfView=9.269,
                            plate_scale=0.0167, 
                            samplingTime=est_tel.samplingTime, 
                            telescope=est_tel, 
                            integrationTime=1./scienceFs, 
                            noiseFlag=False, 
                            logger=test_logger.logger)

    scao_light_path_list = []
    scao_light_path_list.append(LightPath(test_logger.logger))
    # Passed sci=None to avoid sharing the Science Camera instance
    scao_light_path_list[-1].initialize_path(src=sun_red, atm=atm, tel=est_tel, dm=dms[0], wfs=red_wfs, vibration=red_vibrations, sci=None, delay=1)
    
    scao_light_path_list.append(LightPath(test_logger.logger))
    scao_light_path_list[-1].initialize_path(src=ngs_red, atm=atm, tel=est_tel, dm=dms[0], wfs=None, vibration=red_vibrations, sci=red_scicam, delay=1)

    lightPathTasks = [delayed(lp.propagate)(True) for lp in scao_light_path_list]

    test_logger.logger.info(f'Modules initialization took {time.time()-t0} [s]')

    # Interaction Matrix
    t0 = time.time()
    im_handler = InteractionMatrixHandler(test_logger.logger)
    im_handler.initialize_im_class(scao_light_path_list)

    if load_modal_basis and os.path.exists(load_filename_modalBasis):
        im_handler.load_modalBasis(load_filename_modalBasis)
        
    if measure_new_IM or not os.path.exists(load_filename_IM):
        test_logger.logger.warning("Measuring new IM as it was requested or not found.")
        im_handler.measure(modal_basis='kl', stroke=im_stroke, nModes=nModes)
        im_handler.save_IM(save_filename_IM)
        if not (load_modal_basis and os.path.exists(load_filename_modalBasis)):
            im_handler.save_modalBasis(save_filename_modalBasis)
    else:
        im_handler.load_IM(load_filename_IM)

    test_logger.logger.info(f'IM setup took {time.time()-t0} [s]')

    controller_kwargs = {'rcond':0.025, 
                        'beta':1e-4, 
                        'gain':[0.2], 
                        'decay':[0.999], 
                        'ki':[0.5]}
    controller = Controller(telescope=est_tel, 
                            interactionMatrix=im_handler, 
                            reconstructionMethod='tikhonov', 
                            controllerType='leaky', 
                            logger=test_logger.logger, 
                            **controller_kwargs)

    if HAS_TORCH:
        torch.cuda.empty_cache()
    gc.collect()

    test_logger.logger.info('Beginning simulation')

    # SCAO loop
    for i in range(nIterations):
        if (i+1) % 500 == 0 or i == 0:
            est_tel.logger.info(f'Iteration {i+1}/{nIterations}')
        
        atm.update()
        
        # Parallel propagation (n_jobs=1 sequentially avoids sci_cam race conditions/bugs)
        Parallel(n_jobs=1, prefer="threads")(lightPathTasks)
        
        cmd = controller.computeControlAction(scao_light_path_list)
        for j in range(len(dms)):
            dms[j].updateDMShape(cmd[j])

        # Logging peak PSF from the NGS science camera (path index 1)
        test_logger.logger.info(f'PSF peak: {np.max(scao_light_path_list[1].sci_frame)}')
            
        # savepoint.save([atm], i)
        # savepoint.save(dms, i)
        # savepoint.save(scao_light_path_list, i)
        
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