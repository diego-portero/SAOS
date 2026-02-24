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
            rcond : list of length equal to nDMs or float
                Percentage of the maximum singular value below witch the SV are discarded.
            beta : list of length equal to nDMs or float
                Regularisation coefficient beta for the Tikhonov Regularisation: alfa = beta * (Smax**2)
            gain : list of length equal to nDMs or float
                Proportional gain of the Leaky and PI controllers
            decay : list of length equal to nDMs or float
                Decay rate for the Leaky integrator
            ki : list of length equal to nDMs or float
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

        if reconstructionMethod in {'inversion', 'tikhonov'}:
            self.reconstructionMethod = reconstructionMethod
        else:
            self.logger.error('Controller - Unknown reconstructor.')
            raise ValueError('Unknown controller')

        # Default will change to list of size nDMs once the IM is scanned
        self.rcond = kwargs.get('rcond', 0.025)
        self.beta = kwargs.get('beta', 1e-4) # adim, adjusted through trial-error

        # Run the initialization of the reconstructor
        self.reconstructor, self.modal_basis, self.mask, self.altitude = self.initializeReconstructor(self.reconstructionMethod, interactionMatrix)                

        # Setup the controller

        if controllerType in {'leaky', 'forwardPI', 'backwardPI'}:
            self.controllerType = controllerType
        else:
            self.logger.error('Controller - Unknown controller.')
            raise ValueError('Unknown controller')
        
        self.gain = kwargs.get('gain', [0.0 for _ in range(len(self.reconstructor))])
        self.decay = kwargs.get('decay', [0.0 for _ in range(len(self.reconstructor))])
        self.ki = kwargs.get('ki', [0.0 for _ in range(len(self.reconstructor))])

        if not isinstance(self.gain, list):
            temp_gain = self.gain
            self.gain = [temp_gain for _ in range(len(self.reconstructor))]
        if not isinstance(self.decay, list):
            temp_decay = self.decay
            self.decay = [temp_decay for _ in range(len(self.reconstructor))]
        if not isinstance(self.ki, list):
            temp_ki = self.ki
            self.ki = [temp_ki for _ in range(len(self.reconstructor))]                        

        if len(self.gain) != len(self.reconstructor):
            raise ValueError('The gain should be a float or a a list of size nDMs.')
        if len(self.decay) != len(self.reconstructor):
            raise ValueError('The decay should be a float or a a list of size nDMs.')
        if len(self.ki) != len(self.reconstructor):
            raise ValueError('The ki should be a float or a a list of size nDMs.')                

        # Run the initialization of the controller
        self.initializeController(self.controllerType, self.reconstructor)
   
    def initializeReconstructor(self, reconstructionMethod, interactionMatrix):
        self.logger.info('Controller::initializeReconstructor - Computing the reconstructor.')
        t0 = time.time()

        # Define the mask that relates the DMs with the LPs
        
        nDMs = len(interactionMatrix.interaction_matrix_warehouse) # IM warehouse has: nDms x nLPs

        if nDMs < 1:
            raise ValueError('Number of DMs detected are less than 1.')
        
        nLPs = len(interactionMatrix.interaction_matrix_warehouse[0])

        if nLPs < 1:
            raise ValueError('Number of LPs detected are less than 1.')
        
        mask = np.zeros((nDMs, nLPs),dtype=bool)

        # Scan for interactions: if None, then there is not interaction.

        for i in range(nDMs):
            for j in range(nLPs):
                if interactionMatrix.interaction_matrix_warehouse[i][j]['IM'] is not None:
                    mask[i, j] = True

        # Check the reconstructor parameters
        if reconstructionMethod == 'inversion':
            if isinstance(self.rcond, list):
                if len(self.rcond) != nDMs:
                    raise ValueError('Rcond parameter is expected to be a list of size equal to the number of DMs.')
            else:
                # Make the list copying the values
                temp_rcond = self.rcond
                self.rcond = [temp_rcond for _ in range(nDMs)]
        elif reconstructionMethod == 'tikhonov':
            if isinstance(self.beta, list):
                if len(self.rcond) != nDMs:
                    raise ValueError('Beta parameter is expected to be a list of size equal to the number of DMs.')
            else:
                # Make the list copying the values
                temp_beta = self.beta
                self.beta = [temp_beta for _ in range(nDMs)]      
              
        # Get modal basis
        modal_basis = []
        for i in range(nDMs):
            for j in range(nLPs):
                if interactionMatrix.interaction_matrix_warehouse[i][j]['IM'] is not None:
                    # The modal basis is common for each DM
                    modal_basis_type = interactionMatrix.interaction_matrix_warehouse[i][j]['modalBasis']
                    modal_basis.append(torch.as_tensor(interactionMatrix.modal_basis[i][modal_basis_type], dtype=torch.float64, device=self.device))
                    break
        # Get altitudes:
        altitude = []
        for i in range(len(interactionMatrix.dm_scanned_list)):
            altitude.append(interactionMatrix.dm_scanned_list[i].altitude)

        # Now, define the reconstruction matrices for each DM

        reconstructor = []
                
        for i in range(nDMs):
            interaction_matrix_per_DM = []
            for j in range(nLPs):
                if mask[i,j]:
                    # Append the IMs to shape one large matrix of size nValidAct x nSignals
                    interaction_matrix_per_DM.append(interactionMatrix.interaction_matrix_warehouse[i][j]['IM'])
            # Compute the reconstructor
            interaction_matrix_per_DM = torch.as_tensor(np.vstack(interaction_matrix_per_DM), dtype=torch.float64, device=self.device).squeeze()
            if reconstructionMethod == 'inversion':
                temp_reconstructor = torch.linalg.pinv(interaction_matrix_per_DM, self.rcond[i])
            elif reconstructionMethod == 'tikhonov':
                # (D.T@D + alfa*I)@D.T --> implemented through SVD to improve the stability of the inversion and the automation of alfa
                H = interaction_matrix_per_DM
                U, S, Vh = torch.linalg.svd(H, full_matrices=False)
                alfa = self.beta[i] * torch.max(S)**2
                S_reg = S / (S**2 + alfa)
                temp_reconstructor = Vh.T @ torch.diag(S_reg) @ U.T
            else:
                self.logger.error('Controller::initializeReconstructor - Unknown reconstructor')
                raise ValueError('Unknown reconstructor method.')
            reconstructor.append(temp_reconstructor)

        self.logger.info(f'Controller::initializeReconstructor - Reconstruction took {time.time()-t0}[s]')

        return reconstructor, modal_basis, mask, altitude
    
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
            error.append((-1)*torch.as_tensor(np.hstack(combined_slopes).T, dtype=torch.float64, device=self.device).unsqueeze(1)) # -1 for the feedback
        
        # Compute the DM command
        modal_error = []
        modal_cmd = []

        for i in range(len(self.reconstructor)):
            modal_error.append(self.reconstructor[i]@error[i])

            if self.controllerType == 'leaky':
                modal_cmd.append(self.gain[i]*modal_error[i] + self.decay[i] * self.command_previous[i])
            # For the PI (forward and backward), the sampling time is removed from multiplying ki, so that ki is in a closer range to 1, 
            # instead of having large ki values and small proportional gains
            elif self.controllerType == 'forwardPI':
                modal_cmd.append(self.command_previous[i] + self.gain[i] * (modal_error[i]-self.error_previous[i]) + self.ki[i]*self.error_previous[i])
            elif self.controllerType == 'backwardPI':
                modal_cmd.append(self.command_previous[i] + self.gain[i] * (modal_error[i]-self.error_previous[i]) + self.ki[i]*modal_error[i])            

        # Compute the DM command
        dm_cmd = []

        for i in range(len(self.reconstructor)):
            if self.altitude[i] > 0: # TT is discarded automatically in the IM measurement
                dm_cmd.append(self.modal_basis[i][:,2:2+self.reconstructor[i].shape[0]] @ modal_cmd[i])
            else:
                dm_cmd.append(self.modal_basis[i][:,:self.reconstructor[i].shape[0]] @ modal_cmd[i])

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