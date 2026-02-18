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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # Run the initialization of the reconstructor
        self.reconstructor, self.modal_basis, self.mask = self.initializeReconstructor(self.reconstructionMethod, interactionMatrix)
        # Run the initialization of the controller
        self.initializeController(self.controllerType, self.reconstructor)
   
    def initializeReconstructor(self, reconstructionMethod, interactionMatrix):
        self.logger.info('Controller::initializeReconstructor - Computing the reconstructor.')
        t0 = time.time()

        # Define the mask that relates the DMs with the LPs
        
        nDMs = len(interactionMatrix.interaction_matrix_warehouse)

        if nDMs < 1:
            raise ValueError('Number of DMs detected are less than 1.')
        
        nLPs = len(interactionMatrix.interaction_matrix_warehouse)

        if nLPs < 1:
            raise ValueError('Number of LPs detected are less than 1.')
        
        mask = np.zeros((nDMs, nLPs),dtype=bool)

        # Scan for interactions: if None, then there is not interaction.

        for i in range(nDMs):
            for j in range(nLPs):
                if interactionMatrix.interaction_matrix_warehouse[i][j] is not None:
                    mask[i, j] = True

        # Get modal basis
        modal_basis = []
        for i in range(nDMs):
            for j in range(nLPs):
                if interactionMatrix.interaction_matrix_warehouse[i][j] is not None:
                    # The modal basis is common for each DM
                    modal_basis_type = interactionMatrix.interaction_matrix_warehouse[i][j]['modalBasis']
                    modal_basis.append(torch.as_tensor(interactionMatrix.modal_basis[i][modal_basis_type], dtype=torch.float64, device=self.device))
                    break
        # Now, define the reconstruction matrices for each DM

        reconstructor = []
                
        for i in range(nDMs):
            interaction_matrix_per_DM = []
            for j in range(nLPs):
                if mask[i,j]:
                    # Append the IMs to shape one large matrix of size nValidAct x nSignals
                    interaction_matrix_per_DM.append(interactionMatrix.interaction_matrix_warehouse[i][j]['IM'])
            # Compute the reconstructor
            interaction_matrix_per_DM = torch.as_tensor(np.array(interaction_matrix_per_DM), dtype=torch.float64, device=self.device).squeeze()
            if reconstructionMethod == 'inversion':
                temp_reconstructor = torch.linalg.pinv(interaction_matrix_per_DM, self.rcond)
            elif reconstructionMethod == 'tikhonov':
                temp_reconstructor = torch.linalg.inv(interaction_matrix_per_DM.T@interaction_matrix_per_DM + self.alpha*torch.eye(interaction_matrix_per_DM.shape[1]))@interaction_matrix_per_DM.T
            else:
                self.logger.error('Controller::initializeReconstructor - Unknown reconstructor')
                raise ValueError('Unknown reconstructor method.')
            reconstructor.append(temp_reconstructor)

        self.logger.info(f'Controller::initializeReconstructor - Reconstruction took {time.time()-t0}[s]')

        return reconstructor, modal_basis, mask
    
    def initializeController(self, controllerType, reconstructor):

        if controllerType == 'leaky':
            self.command_previous = [torch.zeros((reconstructor[i].shape[0],1), dtype=torch.float64, device=self.device) for i in range(len(reconstructor))]
        elif controllerType == 'forwardPI' or controllerType == 'backwardPI':
            self.command_previous = [torch.zeros((reconstructor[i].shape[0],1), dtype=torch.float64, device=self.device) for i in range(len(reconstructor))]
            self.error_previous = [torch.zeros((reconstructor[i].shape[0],1), dtype=torch.float64, device=self.device) for i in range(len(reconstructor))]
        else:
            self.logger.error('Controller::initializeController - Unknown controller')
            raise ValueError('Unknown controller.')
        return True

    def computeControlAction(self, lightPaths):

        # Get the combined measurement array for each DM
        error = []
        for i in range(len(self.reconstructor)):
            combined_slopes = []

            for j in range(len(lightPaths)):
                if self.mask[i,j]:
                    combined_slopes.append(lightPaths[j].get_wavefront_error())
            
            # Convert to torch
            error.append((-1)*torch.as_tensor(np.array(combined_slopes).T, dtype=torch.float64, device=self.device)) # -1 for the feedback
        
        # Compute the DM command
        modal_error = []
        modal_cmd = []

        for i in range(len(self.reconstructor)):
            modal_error.append(self.reconstructor[i]@error[i])

            if self.controllerType == 'leaky':
                modal_cmd.append(self.gain*modal_error[i] + self.decay * self.command_previous[i])
            elif self.controllerType == 'forwardPI':
                modal_cmd.append(self.command_previous[i] + self.gain * (modal_error[i]-self.error_previous[i]) + self.ki*self.samplingTime*self.error_previous[i])
            elif self.controllerType == 'backwardPI':
                modal_cmd.append(self.command_previous[i] + self.gain * (modal_error[i]-self.error_previous[i]) + self.ki*self.samplingTime*modal_error[i])            

        # Compute the DM command
        dm_cmd = []

        for i in range(len(self.reconstructor)):
            dm_cmd.append(self.modal_basis[i] @ modal_cmd[i])

        # Update history buffers for the next iteration

        if self.controllerType == 'leaky':
            self.command_previous = modal_cmd.copy()
        elif self.controllerType == 'forwardPI' or self.controllerType == 'backwardPI':
            self.command_previous = modal_cmd.copy()
            self.error_previous = modal_error.copy()

        return dm_cmd

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