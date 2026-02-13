import numpy as np

import h5py
import cv2

import logging
import logging.handlers
from queue import Queue

class NCPA:
    def __init__(self,
                 telescope,
                 logger = None):
        """
        Initialize a NCPA generator.

        Parameters
        ----------
        telescope : Telescope
            Telescope object to which the WFS is attached.
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
        self.tag = 'ncpa'
        
        # Generate empty OPD

        self.ncpa_opd = np.zeros_like(telescope.pupil, dtype=np.float32)


    def load(self, filename): 
        """
        Load the NCPA OPD from a H5 file.

        Parameters
        ----------
        filename : str
            Path and base filename (with extension) of the H5 file to load.

        Returns
        -------
        bool
            True if loaded successfully.
        """
        self.logger.debug('NCPA::load')


        if not filename.endswith(".h5"):
            filename += ".h5"        

        with h5py.File(filename, 'r') as f:
            temp_opd = f['opd']['data'][()]

            self.logger.info(f"NCPA::load - Loaded OPD.")     

        if (temp_opd.shape[0] != self.ncpa_opd.shape[0]) or (temp_opd.shape[1] != self.ncpa_opd.shape[1]):
            self.logger.warning('NCPA::Load - The dimensions of the NCPA do not match the telescope\'s pupil. Interpolating, may introduce artifacs.')
            self.ncpa_opd = cv2.resize(temp_opd, (self.ncpa_opd.shape[0], self.ncpa_opd.shape[1]), interpolation=cv2.INTER_LINEAR)

        return True
    
    def setOPD(self, input_opd):
        """
        set the NCPA OPD directly from a variable

        Parameters
        ----------
        input_opd : np.ndarray
            Input OPD in [m]

        Returns
        -------
        bool
            True if loaded successfully,.
        """
        self.logger.debug('NCPA::load')

        if (input_opd.shape[0] != self.ncpa_opd.shape[0]) or (input_opd.shape[1] != self.ncpa_opd.shape[1]):
            self.logger.error('NCPA::setOPD - The dimensions of the NCPA do not match the telescope\'s pupil. Rejecting.')
            raise ValueError('The dimensions of the NCPA do not match the telescope\'s pupil.')

        self.ncpa_opd = input_opd.copy()
        
        return True
    
    def getPhase(self):
        """
        Returns the NCPA OPD.

        Parameters
        ----------
        Returns
        -------
        np.ndarray
            NCPA OPD [m]
        """
        self.logger.debug('NCPA::getPhase')
       
        return self.ncpa_opd
    
    def getCurrentVibrations(self, iteration):
        """
        Return the vibrations for the iteration asked.

        Parameters
        ----------
        iteration : int
            Iteration of the AO simulation. If iteration is larger than
            the temporal sequence of the vibrations, the method wraps-around.

        Returns
        -------
        np.ndarray
           Vibration combining X-Y vibrations for the given iteration
        """        
        if (iteration > len(self.x_vibrations_stroke)) or (iteration > len(self.x_vibrations_stroke)):
            self.logger.warning('Vibration::getCurrentVibrations - The length of the vibrations array is smaller than the simulation window. Wrapping-around.')
            iteration = iteration % np.minimum(len(self.x_vibrations_stroke), len(self.y_vibrations_stroke))
        
        opd_vibrations = self.x_vibrations_stroke[iteration]*self.x_mode + self.y_vibrations_stroke[iteration]*self.y_mode
        
        return opd_vibrations

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
