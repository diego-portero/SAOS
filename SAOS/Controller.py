import numpy as np
import torch

import h5py
import time

import logging
import logging.handlers
from queue import Queue

class Controller:
    def __init__(self,
                 telescope,
                 interactionMatrix,
                 controllerType,
                 reconstructionMethod,                 
                 logger = None,
                 **kwargs):
        """
        Initialize the Controller module.

        Parameters
        ----------
        telescope : Telescope instance
            Telescope instance, provides the sampling time of the simulation.
        interactionMatrix : InteractionMatrixHandler instance
            Contains the interaction matrices and modal basis for the simulation configuration.
        controllerType : String
            The type of controller that will be used, supported types are: {leaky, forwardPI, backwardPI}. 
        reconstructionMethod : String
            Type of reconstructor used, supported types are: {inversion, tikhonov}.        
        **kwargs
            rcond : float
                Percentage of the maximum singular value below witch the SV are discarded.
            alpha : float
                Regularisation coefficient alfa for the Tikhonov Regularisation
            gain : float
                Proportional gain of the Leaky and PI controllers
            decay : float
                Decay rate for the Leaky integrator
            ki : float
                Integral gain for the PI controllers            
        """        
        # Setup the logger to handle the queue of info, warning and errors msgs in the simulator
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
            self.external_logger_flag = False
        else:
            self.external_logger_flag = True
            self.logger = logger

        # Define class attributes
        self.tag = 'controller'

        self.samplingTime = telescope.samplingTime

        if controllerType in {'leaky', 'forwardPI', 'backwardPI'}:
            self.controllerType = controllerType
        else:
            self.logger.error('Controller - Unknown controller.')
            raise ValueError('Unknown controller')
        
        if reconstructionMethod in {'inversion', 'tikhonov'}:
            self.reconstructionMethod = reconstructionMethod
        else:
            self.logger.error('Controller - Unknown reconstructor.')
            raise ValueError('Unknown controller')
        
        self.rcond = kwargs.get('rcond', 0.025)
        self.alpha = kwargs.get('alpha', 0.03) # Gonzalez-Cava et al. (2022), Laboratory Results of SCAO: getting ready for the EST MCAO

        self.gain = kwargs.get('gain', 0.0)
        self.decay = kwargs.get('decay', 0.0)
        self.ki = kwargs.get('ki', 0.0)

        # Run the initialization of the controller
        self.initializeController(self.controllerType)
        # Run the initialization of the reconstructor
        self.reconstructor, self.mask = self.initializeReconstructor(self.reconstructionMethod, interactionMatrix)        

    def initializeController(self, controllerType):

        return True
    
    def initializeReconstructor(self, reconstructionMethod, interactionMatrix):
        self.logger.info('Controller::initializeReconstructor - Computing the reconstructor.')
        t0 = time.time()

        # Define the mask that relates the DMs with the LPs
        
        nDMs = len(interactionMatrix.interaction_matrix_warehouse)

        if nDMs < 1:
            raise ValueError('Number of DMs detected are less than 0.')
        
        nLPs = len(interactionMatrix.interaction_matrix_warehouse)

        if nLPs < 1:
            raise ValueError('Number of LPs detected are less than 0.')
        
        mask = np.zeros((nDMs, nLPs),dtype=bool)

        # Scan for interactions: if None, then there is not interaction.

        for i in range(nDMs):
            for j in range(nLPs):
                if interactionMatrix.interaction_matrix_warehouse[i][j] is not None:
                    mask[i, j] = True
                
        reconstructor = None
        mask = None

        self.logger.info(f'Controller::initializeReconstructor - Reconstruction took {time.time()-t0}[s]')

        return reconstructor, mask

    def computeControlAction(self, lightPaths):
        return True

    def setup_logging(self, logging_level=logging.WARNING):
        #
        #  Setup of logging at the main process using QueueHandler
        log_queue = Queue()
        queue_handler = logging.handlers.QueueHandler(log_queue)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging_level)  # Minimum log level

        # Setup of the formatting
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )

        # Output to terminal
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Qeue handler captures the messages from the different logs and serialize them
        queue_listener = logging.handlers.QueueListener(log_queue, console_handler)
        root_logger.addHandler(queue_handler)
        queue_listener.start()

        return queue_listener
    
    # The logging Queue requires to stop the listener to avoid having an unfinalized execution. 
    # If the logger is external, then the queue is stop outside of the class scope and we shall
    # avoid to attempt its destruction
    def __del__(self):
        if not self.external_logger_flag:
            self.queue_listerner.stop()