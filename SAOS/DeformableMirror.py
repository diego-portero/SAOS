# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:32:10 2020

@author: cheritie

Major update on March 24 2025
@author: nrodlin
"""


import numpy as np
import torch
import scipy as sp
from joblib import Parallel, delayed

import logging
import logging.handlers
from queue import Queue

from .MisRegistration import MisRegistration
from .tools.interpolateGeometricalTransformation import interpolate_cube
from .tools.tools import pol2cart


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLASS INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

class dmLayerClass():
    def __init__(self):
        self.altitude = None
        self.D_fov                  = None # diameter of the DM projected into the altitude layer [meters]
        self.D_px                   = None # size of the DM in [pixels]
        self.telescope_D            = None # Telescope diamter in [meters]
        self.telescope_resolution   = None # Telescope diameter in [px] using the original telescope resolution
        self.center                 = None # center coordinates of the DM [pixels]
        self.metapupil              = None # 2D telescope pupil at the DM altitude [circular mask], at 0km equals the pupil without spider nor central obs
        self.pupil                  = None # 2D telescope pupil [binary mask]
        self.OPD                    = None # stores the layer OPD without projection to any source (full pupil/metapupil)
        self.cmd_1D                 = None # stores the 1D DM command, including valid and invalid actuators
        
"""
Deformable Mirror Module
=================

This module contains the `DeformableMirror` class, used for modeling a deformable mirror in adaptive optics simulations.
"""

class DeformableMirror:
    def __init__(self,
                 telescope,
                 nActs:float,
                 mechCoupling:float = 0.35,
                 coordinates:np.ndarray = None,
                 pitch:float = None,
                 modes:np.ndarray = None,
                 misReg = None,
                 typeDM:str = 'cartesian',
                 floating_precision:int = 64,
                 altitude:float = None,
                 flip = False,
                 flip_lr = False,
                 sign = 1,
                 valid_act_thresh_outer = None,
                 logger = None,
                 **kwargs):
        """
        Initialize a Deformable Mirror (DM) with zonal or modal influence functions.

        Parameters
        ----------
        telescope : Telescope
            Telescope associated with this DM.
        nActs : float
            Number of actuators in the horizontal axis of the pupil.
        mechCoupling : float, optional
            Coupling factor between actuators, by default 0.35.
        coordinates : np.ndarray, optional
            Custom actuator coordinates.
        pitch : float, optional
            Actuator pitch in meters.
        modes : np.ndarray, optional
            Influence functions or modal basis.
        misReg : MisRegistration, optional
            Misregistration object for geometrical offsets.
        typeDM : str, optional
            Type of the DM: {cartesian, radial, custom}. By default, custom.
        floating_precision : int, optional
            Use 32 or 64-bit floats, by default 64.
        altitude : float, optional
            Conjugation altitude of the DM in meters.
        flip : bool, optional
            Flip the influence functions vertically.
        flip_lr : bool, optional
            Flip the influence functions left-right.
        sign : int, optional
            Sign of actuation.
        valid_act_thresh_outer : float, optional
            Threshold for validating actuators outside pupil.
        logger : logging.Logger, optional
            Logger instance.
        **kwargs : dict, optional
            Additional keyword arguments.
            
            validActThreshpercentage : float, optional
                Parameter to select a percentage of the actuator pitch to consider it valid o not.
            maxStrokePtV : float, optional
                Maximum mechanical stroke peak-to-valley in [m]. By default 100e-6 [m].
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
        self.tag = 'deformableMirror'

        self.floating_precision = floating_precision
        self.flip_= flip
        self.flip_lr = flip_lr 
        self.sign = sign
        self.altitude = altitude
        self.nActs = nActs

        if mechCoupling <=0:
            raise ValueError('The value of mechanical coupling should be positive.')
        else:
            self.mechCoupling          = mechCoupling
        
        # Define the DM layer       
        self.dm_layer = self.buildLayer(telescope, altitude)

        if pitch is None:
            self.pitch = self.dm_layer.D_fov/(nActs-1)  # size of a subaperture
        else:
            self.pitch = pitch
        
        if misReg is None:
            # create a MisReg object to store the different mis-registration
            self.misReg = MisRegistration(0,0,0,1,telescope=telescope, logger=self.logger)
        else:
            self.misReg=misReg            

        self.valid_act_thresh_outer = valid_act_thresh_outer
        self.validActThreshpercentage = kwargs.get('validActThreshpercentage', 0.7533)
        self.maxStrokePtV = kwargs.get('maxStrokePtV', 100e-6) # [m]      

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Depending on the type of the DM, the coordinates are generated differently
        if typeDM == 'cartesian':
            # Define the coordinates
            self.coordinates, self.validAct, self.nValidAct = self.generate_cartesian_dm(nActs)
        elif typeDM == 'radial':
            self.logger.warning('DeformableMirror::__init__ - Radial DM is not yet supported in this new version, using default.')
            self.typeDM = 'cartesian'
        elif typeDM == 'custom':
            self.logger.warning('DeformableMirror::__init__ - Custon DM is not yet supported in this new version, using default.')
            self.typeDM = 'cartesian'
        else:
            self.logger.error('DeformableMirror::__init__ - Unrecognized DM type, using default. Implemented are: [cartesian, radial, custom]')
            raise ValueError('Unrecognized DM type, using default.')
        
        # Compute the interpolation weights
        self.interp_weights, self.interp_idx = self.high_res_interpolator(self.coordinates, self.validAct) # Shepard-kind interpolation
    
    # Generation of the cartesian coordinates and mask of valid actuators using the outer pupil limit
    
    def generate_cartesian_dm(self, nActs):
        """
        Generates a distribution of cartesian points and a logic mask 
        filtering the points that are within the limits of the external
        pupil diameter.

        Parameters
        ----------
        nActs : int
            Number of actuators in the square side

        Returns
        -------
        coordinates : numpy.ndarray
            X and Y coordinates aranged as [nActs**2,2]
        validAct : numpy.ndarray
            Logic mask of valid actuators
        nValidAct : int
            Number of valid actuators
        """
        # First, we need to generate the coordinates for the cartesian grid, centered in the optical axis.

        x    = np.linspace(-(self.dm_layer.D_fov)/2,(self.dm_layer.D_fov)/2, nActs)
        X, Y = np.meshgrid(x,x)

        coordinates = np.array([X.flatten(), Y.flatten()]).T

        # Second, define mask to obtain the valid actuators

        r = np.sqrt(X**2 + Y**2)
        
        if self.valid_act_thresh_outer is None:
            self.valid_act_thresh_outer = self.dm_layer.D_fov/2#+self.validActThreshpercentage*self.pitch
        
        validAct  = r <= self.valid_act_thresh_outer
        nValidAct = np.sum(validAct)

        return coordinates, validAct.flatten(), nValidAct

    # Interpolates from a set of coordinates without requiring a specific
    # distribution to a grid of points.

    def high_res_interpolator(self, coordinates, validActuators, k_nearest=24):
        """
        Generates a grid of points from a set of points distributed as indicated
        by the coordinates matrix, without requiring any specific distribution. Uses a Shepard-kind interpolation.

        Parameters
        ----------
        coordinates : numpy.ndarray
            X and Y coordinates aranged as [nActs**2,2]
        validAct : numpy.ndarray
            Logic mask of valid actuators
        k_nearest : optional, int            
            Selects the k-nearest neighbourgs. By default 8.

        Returns
        -------
        weights : torch.Tensor
            Weighting matrix to perform the interpolation using the valid actuators
        idx : torch.Tensor
            Indices to select the k-neighbours of each point
        """
        # Convert from ndarray to tensors for performance

        coordinates_torch = torch.as_tensor(coordinates[validActuators], dtype=torch.float32, device=self.device)

        # Make a grid of points

        x    = np.linspace(-(self.dm_layer.D_fov)/2,(self.dm_layer.D_fov)/2, self.dm_layer.D_px)
        grid_x, grid_y = np.meshgrid(x,x)

        gx = torch.as_tensor(grid_x, dtype=torch.float32, device=self.device).reshape(-1)
        gy = torch.as_tensor(grid_y, dtype=torch.float32, device=self.device).reshape(-1)
        grid_highres = torch.stack([gx, gy], dim=1)  # (D_px**2,2)

        # Compute the distance between each point in the grid and the points defined by the input coordinates matrix
        d = torch.cdist(grid_highres, coordinates_torch, p=2.0) # distance using 2-norm (euclidean distance)
        d2 = d * d # compute the square of the distance

        # Select k-neighbours
        d2k, idx = torch.topk(d2, k=k_nearest, dim=1, largest=False, sorted=False)

        # Select limit distance analysing the neighbourhood
        # distance to the farest neighbour
        dk = torch.sqrt(d2k.max(dim=1).values)
        ls = dk.median().clamp_min(1e-6)
        lengthscale = float(ls.item())
        
        inv_l2 = 1.0 / (lengthscale * lengthscale)

        # Compute Gaussian kernel for the interpolation

        weights = torch.exp(-0.5 * d2k * inv_l2)

        # Normalize the weights

        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

        return weights, idx.to(torch.int64)
    
    # The DM can be considered as an atmospheric layers with discrete points actuated, which are then connected with their influence functions, 
    # shaping a continuous 2D surface. 
    def buildLayer(self, telescope, altitude):
        """
        Construct and configure the DM layer at a given conjugation altitude.

        Parameters
        ----------
        telescope : Telescope
            Telescope providing aperture and resolution information.
        altitude : float
            Altitude in meters to conjugate the DM layer.

        Returns
        -------
        dmLayerClass
            Configured DM layer with geometric and aperture metadata.
        """
        self.logger.debug('DeformableMirror::buildLayer')
        # initialize layer object
        layer                   = dmLayerClass()
       
        # gather properties of the atmosphere
        if altitude is None:
            layer.altitude          = 0
        else:
            layer.altitude          = altitude
                
        # Diameter and resolution of the layer including the Field Of View and the number of extra pixels
        layer.D_fov             = telescope.D + 2*np.tan(telescope.fov/(206624*2))*layer.altitude # in [m]
        layer.D_px              = int(np.ceil((telescope.resolution/telescope.D)*layer.D_fov)) # Diameter in [px]
        layer.center            = layer.D_px//2

        layer.OPD               = np.zeros([layer.D_px,layer.D_px]) # stores the layer OPD without projection to any source (full pupil/metapupil)
        layer.cmd_1D            = None # stores the 1D DM command, including valid and invalid actuators

        layer.telescope_D          = telescope.D # Telescope diameter in [m]
        layer.telescope_resolution = telescope.resolution # Telescope diameter in [px] using the original telescope resolution

        # Circular entrance pupil
        x = np.linspace(-layer.D_px/2, layer.D_px/2, layer.D_px)
        xx, yy = np.meshgrid(x, x)
        layer.metapupil = xx**2 + yy**2 < ((layer.D_px + 1)/2)**2
        layer.pupil                 = telescope.pupil.copy()
        
        return layer
    # When the DM is located at an altitude layer, the phase of the DMs affect differently the sources depending on their coordinates in sky. 
    # Before return the phase of the DM, we need to select the correct region of the DM affecting the source, which is done by masking an square area
    def get_dm_pupil(self, src):
        """
        Compute pupil mask seen by a source at the DM altitude.

        Parameters
        ----------
        src : Source
            Source object with angular position.

        Returns
        -------
        np.ndarray
            Binary square mask (1s where the source is affected).
        """
        self.logger.debug('DeformableMirror::get_dm_pupil')
        
        # Source coordinates are [angle_fov["], zenith_angle[rad]]. Hence, to obtain the location of the object at the DM altitude plane:
        # 1) Compute the projection: altitude * tan(angle_fov[rad]) -> location in meters
        # 2) From meters to pixels: result_1 * (D_px/metapupil_D)
        # 3) From polar to cartesian: (result_2[px], zenith_angle[rad]) -> (x_z, y_z) [px]
        [x_z, y_z] = pol2cart(self.dm_layer.altitude * np.tan(src.coordinates[0]/206265)*(self.dm_layer.D_px/self.dm_layer.D_fov), 
                             np.deg2rad(src.coordinates[1]))

        # Matriz origin is placed at the left-top corner, whereas the telescope origin is at the optical axis.
        # We add an offset to translate the origins.
        center_x = int(y_z) + self.dm_layer.D_px//2
        center_y = int(x_z) + self.dm_layer.D_px//2
    
        # Finally, we mask the region that sees the source. This region is centered at the location computed in 3) 
        # and its shape equals the telescope pupil with the DM layer diameter in [px]
        square_mask = np.zeros([self.dm_layer.D_px, self.dm_layer.D_px])
        # Define square limits to take the region of the metapupil affecting the source
        left_corner_x = center_x-self.dm_layer.telescope_resolution//2
        left_corner_y = center_y-self.dm_layer.telescope_resolution//2
        # Mask the region
        square_mask[left_corner_x:left_corner_x + self.dm_layer.telescope_resolution,
                    left_corner_y:left_corner_y + self.dm_layer.telescope_resolution] = 1
        
        return square_mask

    # The OPD is computed for a given source - for altitude layers, it depends on its location in sky
    # Returns the OPD [m] and the phase [rad], for which the wavelength of the input source is used. 
    # The shape of the output is [telescope.resolution, telescope.resolution] [px]
    def get_dm_opd(self, source):
        """
        Compute the Optical Path Difference (OPD) and phase from the DM for a given source.

        Parameters
        ----------
        source : Source
            Source object defining wavelength and position.

        Returns
        -------
        tuple of np.ndarray
            OPD in meters and phase in radians.
        """
        self.logger.debug('DeformableMirror::get_dm_opd')
        # Get the pupil for the object. For the case of the sun, only the central subdir is considered.
        pupil = self.get_dm_pupil(source) 
        # Apply mis-registration
        opd_misregistered = self.misReg.apply_misreg(self.dm_layer.OPD)
        # Select only the region of the DM that is affecting to the source.
        OPD = np.zeros([self.dm_layer.telescope_resolution, self.dm_layer.telescope_resolution])
        OPD = opd_misregistered[pupil==1].reshape(self.dm_layer.telescope_resolution, self.dm_layer.telescope_resolution)
        # Depending on the source type, certain action may differ
        if source.tag == 'LGS':
            # This code considers the impact of having an object at a finite altitude (typ. LGS). 
            sub_im = np.atleast_3d(OPD)
                
            alpha_cone = np.arctan(self.dm_layer.telescope_D/2/source.altitude)
            h = source.altitude-self.dm_layer.altitude

            if np.isinf(h):
                r = self.dm_layer.telescope_D/2
            else:
                r = h*np.tan(alpha_cone)

            ratio = self.dm_layer.telescope_D/r/2
            
            cube_in = sub_im.T
            pixel_size_in   = self.dm_layer.D_fov/self.dm_layer.D_px
            pixel_size_out  = pixel_size_in/ratio
            
            output_OPD = np.asarray(np.squeeze(interpolate_cube(cube_in, pixel_size_in, pixel_size_out, self.dm_layer.telescope_resolution)))
        
        else: # NGS and Sun types can be handled equally. The sun is simplified, only considering the projection of the centrar subdir
            output_OPD = OPD * self.dm_layer.pupil

        output_phase = output_OPD * (2*np.pi / source.wavelength)       

        return output_OPD, output_phase

    # Saturate the actuation of the deformable mirror

    def saturateShape(self, cmd):
        """
        Saturate the command of the mirror.

        Parameters
        ----------
        cmd : np.ndarray
            Command required from the mirror.

        Returns
        -------
        cmd_saturated
            Command executed by the mirror.
        """
        self.logger.debug('DeformableMirror::saturateShape') 

        # The OPD is treated in the mulator as wavefront --> the PtV maximum is equivalent to the wavefront value
        cmd_saturated = np.clip(cmd, a_min=-self.maxStrokePtV, a_max=self.maxStrokePtV)
        return cmd_saturated

    # The shape of the mirror is controlled through a set of modes that by default are zonal --> defining a typical DM. 
    # If a modal DM is defined, then the coefficients correspond to those of the modal basis.
    # Please notice that in this context, the modes do not refer to the AO control modal base but the intrinsic mechanical behaviour of the DM.
    # The shape of the mirror is computed as the matricial product of modes x coeffs -> modes [dm_layer.D_px, nValidActs], coefs [nValidActs, 1]    
    def updateDMShape(self, val):
        """
        Update the OPD map from the current coefficients or 2D grid.

        Parameters
        ----------
        val : np.ndarray
            Either a coefficient vector or a 2D shape map.

        Returns
        -------
        bool
            True if update was successful.
        """
        self.logger.debug('DeformableMirror::updateDMShape') 

        if isinstance(val, np.ndarray):
            if val.ndim > 1:
                self.logger.error(f'DeformableMirror::updateDMShape - Shape of the command is not supported. Expected 1D array.')                
                raise ValueError('Shape of the command is not supported. Expected 1D array.')

            if val.shape[0] == self.validAct.shape[0]:
                # Command received is 1D, without filtering the unused actuators
                val = val[self.validAct]
            elif val.shape[0] == self.nValidAct:
                # Command received is 1D, only valid actuators
                val = val
            else:
                self.logger.error(f'DeformableMirror::updateDMShape - Size of the command is not correct: {val.shape}.')
                raise ValueError('Size of the command is not correct.')
        
        # Fill the layer 1D command
        temp = np.zeros_like(self.validAct, dtype=val.dtype)
        temp[self.validAct] = val
        
        self.dm_layer.cmd_1D = temp.copy()

        # Apply the Shepard interpolator, with the pre-computed weights
        coefs_torch           = torch.as_tensor(val, dtype=torch.float32, device=self.device)
        coefs_neighbourhood   = coefs_torch[self.interp_idx]
        opd_highres           = (self.interp_weights * coefs_neighbourhood).sum(dim=1)

        self.dm_layer.OPD     = opd_highres.cpu().numpy().reshape(self.dm_layer.D_px, self.dm_layer.D_px)

        # Saturate the actuation
        self.dm_layer.OPD = self.saturateShape(self.dm_layer.OPD)

        return True
    
    def updateMisreg(self, elapsedTime):
        """
        Update the mis-registration params by the temporal factor
        Returns
        --------
        True
        """
        self.misReg.update_params(elapsedTime)

        return True
            
    def print_properties(self):
        """
        Print a summary of the DM configuration.

        Returns
        -------
        None
        """
        self.logger.info('DeformableMirror::print_properties')
        self.logger.info('DeformableMirror::print_properties')
        self.logger.info('{: ^21s}'.format('Controlled Actuators')                     + '{: ^18s}'.format(str(self.nValidAct)))
        self.logger.info('{: ^21s}'.format('Pitch')                                    + '{: ^18s}'.format(str(self.pitch))                    +'{: ^18s}'.format('[m]'))
        self.logger.info('{: ^21s}'.format('Mechanical Coupling')                      + '{: ^18s}'.format(str(self.mechCoupling))             +'{: ^18s}'.format('[%]' ))
        self.logger.info('Mis-registration:')
        self.misReg.print_properties()

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
