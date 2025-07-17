"""
Created on July 10 2025
@author: nrodlin and adrianc
"""
import time
import zmq
import pickle

import logging
import logging.handlers
from queue import Queue

"""
ReceptionPoint Module
=================

This module contains the `ReceptionPoint` class, used to ask for specific buffer to an external server
"""

class ReceptionPoint:
    def __init__(self, logger=None, 
                 port=7000, 
                 ip="161.72.210.177", 
                 protocol="tcp", 
                 timeout=5000):
        """
        Initialize the Sharepoint publisher for sharing light path data.

        Parameters
        ----------
        logger : logging.Logger, optional
            External logger to use. If None, initializes internal logging.
        port : int, optional
            Port number for ZeroMQ publisher. Default is 5555.
        ip : str, optional
            IP address to bind the publisher. Default is localhost.
        protocol : str, optional
            Communication protocol (e.g., 'tcp').
        timeout : int
            Timeout for the response to arrive in ms.       
        """
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
        else:
            self.external_logger_flag = True
            self.logger = logger
            
        # Map arguments to attributes
        self.ip = ip
        self.port = port
        self.protocol = protocol
        self.timeout = timeout
        # Setup connecction
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)

        self.socket.connect(self.protocol + '://' + self.ip + ':' + str(self.port))
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
	
    def sendRequest(self, buffer_type=''):
        """
        Send a request to a ZMQ server for an specific answer, provided
        by the command type.
		
        Parameters
        --------------------
        buffer_type : str
            Empty string by default, shall contain a specific buffer type
        Returns
        --------------------
        Value of the response, if None then an error occurred. 
        """
		
        result = None
		
        if buffer_type == 'actuator_cmd':
            while True:
                self.socket.send(pickle.dumps(buffer_type))
                self.logger.info('ReceptionPoint::sendRequest - request sent. Waiting for answer')

                if self.poller.poll(timeout=self.timeout):
                    data = self.socket.recv()
                    response = pickle.loads(data)
                    self.logger.info('ReceptionPoint::sendRequest - answer received.')
                    result = response
                    break
                else:
                    result = None
                    
                    # Reset socket
                    try:
                        self.socket.setsockopt(zmq.LINGER,0)
                        self.socket.close()
                    except Exception:
                        pass
					
                    self.socket = self.context.socket(zmq.REQ)
                    self.socket.connect(self.protocol + '://' + self.ip + ':' + str(self.port))
                    
                    self.poller.register(self.socket, zmq.POLLIN)
                    self.logger.error('ReceptionPoint::sendRequest - Timeout.')
                    break
        else:
            result = None
            self.logger.warning('ReceptionPoint::sendRequest - Unsupported buffer type.')
        return result
    
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
    
    def __del__(self):
        if not self.external_logger_flag:
            self.queue_listerner.stop()
        if self.socket is not None and self.context is not None:
            self.socket.close()
            self.context.term()
