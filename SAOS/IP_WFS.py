# -*- coding: utf-8 -*-
"""
Created on Thu May 8 2025

@author: dportero
"""

import logging
import numpy as np
import torch
import matplotlib.pyplot as plt

from skimage.restoration import unwrap_phase
from scipy.interpolate import RBFInterpolator


class IP_WFS:
    def __init__(
        self,
        nSubap,
        telescope,
        src,
        OPD,
        lightRatio,
        plate_scale,
        fieldOfView,
        guardPx,
        logger=None,
        **kwargs
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.tag = "IP-WFS"

        # Basic parameters
        self.nSubap = nSubap
        self.telescope = telescope
        self.src = src
        self.OPD = OPD
        self.resolution = telescope.resolution

        self.lightRatio = lightRatio
        self.plate_scale = plate_scale
        self.fieldOfView = fieldOfView
        self.guardPx = guardPx
        self.unit_in_rad = kwargs.get("unit_in_rad", True)

        self.sensing_wavelength = kwargs.get(
            "sensing_wavelength",
            680e-9
        )
        self.science_wavelength = kwargs.get(
            "science_wavelength",
            self.sensing_wavelength
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Geometry
        self.subaperture_size = telescope.D / nSubap
        self.npix_phase = telescope.resolution // nSubap

        # Valid subapertures
        interp = torch.nn.functional.interpolate(
            torch.from_numpy(telescope.pupil.astype(float))
            .unsqueeze(0)
            .unsqueeze(0),
            size=(nSubap * self.npix_phase, nSubap * self.npix_phase),
            mode="bilinear",
            align_corners=True
        )

        self.pupil_interpolation_mask = interp.squeeze().numpy()

        self.valid_subapertures = np.zeros(
            (nSubap, nSubap),
            dtype=bool
        )

        for i in range(nSubap):
            for j in range(nSubap):
                block = self.pupil_interpolation_mask[
                    i * self.npix_phase:(i + 1) * self.npix_phase,
                    j * self.npix_phase:(j + 1) * self.npix_phase
                ]

                if np.mean(block) > 0.95:
                    self.valid_subapertures[i, j] = True

        self.nValidSubaperture = int(
            np.sum(self.valid_subapertures)
        )

        # OOPAO compatibility
        self.signal = None
        self.signal_2D = None
        self.cam = None
        self.nSignal = self.nValidSubaperture

        self.debug = {}

        self._init_lens_geometry()

        self.logger.info("IP-WFS initialized")

    # ============================================================
    # Lens geometry
    # ============================================================
    def _init_lens_geometry(self):
        N = self.resolution
        num = self.nSubap

        self.lens_radius = N // (2 * num)
        self.center = N // 2

        spacing = 2 * self.lens_radius
        offsets = (
            np.arange(num) - (num - 1) / 2
        ) * spacing

        y, x = np.ogrid[:N, :N]

        masks = []
        for dx in offsets:
            for dy in offsets:
                mask = (
                    (x - self.center - dx) ** 2
                    + (y - self.center - dy) ** 2
                    <= self.lens_radius ** 2
                )
                masks.append(mask)

        self.lens_masks = np.array(masks)

    # ============================================================
    # Interference model
    # ============================================================
    def compute_interference(
        self,
        lens_grid,
        phase_offset=0.0
    ):
        num = lens_grid.shape[0]
        output = np.zeros((num, num))

        snake_indices = []
        for i in range(num):
            row = list(range(num))
            if i % 2 == 1:
                row = row[::-1]
            for j in row:
                snake_indices.append((i, j))

        for k in range(len(snake_indices) - 1):
            i1, j1 = snake_indices[k]
            i2, j2 = snake_indices[k + 1]

            phi1 = (
                2 * np.pi * lens_grid[i1, j1] * 1e-9
                / self.sensing_wavelength
            )
            phi2 = (
                2 * np.pi * lens_grid[i2, j2] * 1e-9
                / self.sensing_wavelength
            )

            output[i2, j2] = 2 * (
                1 + np.cos(phi1 - phi2 + phase_offset)
            )

        return output

    # ============================================================
    # Phase extraction
    # ============================================================
    def extract_phase(self, I1, I2):
        eps = 1e-9

        arg1 = np.clip(
            np.rot90(I1) / 2 - 1,
            -1 + eps,
            1 - eps
        )
        arg2 = np.clip(
            np.rot90(I2) / 2 - 1,
            -1 + eps,
            1 - eps
        )

        phase1 = np.arccos(arg1)
        phase2 = np.arccos(arg2)

        return np.sign(phase1 - phase2) * phase1

    # ============================================================
    # Snake integration
    # ============================================================
    def integrate_snake(self, matrix):
        cumulative = np.zeros_like(matrix)

        snake_indices = []
        for i in range(self.nSubap):
            row = list(range(self.nSubap))
            if i % 2 == 1:
                row = row[::-1]
            for j in row:
                snake_indices.append((i, j))

        rot = np.rot90(matrix, 3)

        total = 0.0
        for i, j in snake_indices:
            total += rot[i, j]
            cumulative[i, j] = total

        return cumulative

    # ============================================================
    # Interpolation
    # ============================================================
    def interpolate_surface(self, coarse_phase):
        y, x = np.indices(coarse_phase.shape)

        points = np.column_stack((
            x[self.valid_subapertures],
            y[self.valid_subapertures]
        ))

        values = coarse_phase[self.valid_subapertures]

        yy, xx = np.indices(
            (self.resolution, self.resolution)
        )

        xx_scaled = (
            xx * (self.nSubap - 1)
            / (self.resolution - 1)
        )
        yy_scaled = (
            yy * (self.nSubap - 1)
            / (self.resolution - 1)
        )

        query = np.column_stack((
            xx_scaled.ravel(),
            yy_scaled.ravel()
        ))

        rbf = RBFInterpolator(
            points,
            values,
            kernel="thin_plate_spline"
        )

        interp = rbf(query).reshape(
            self.resolution,
            self.resolution
        )

        interp *= self.telescope.pupil

        return interp

    # ============================================================
    # Main WFS
    # ============================================================
    def wfs_measure(self, OPD, src=None):
        if OPD.ndim == 3:
            OPD = OPD[OPD.shape[0] // 2]

        if OPD.ndim != 2:
            raise ValueError(
                f"Expected 2D OPD, got {OPD.shape}"
            )

        # Lens averaging
        lens_means = np.array([
            np.mean(OPD[mask])
            for mask in self.lens_masks
        ])

        lens_grid = lens_means.reshape(
            self.nSubap,
            self.nSubap
        )

        # Interference
        I1 = self.compute_interference(
            lens_grid,
            phase_offset=0.0
        )

        I2 = self.compute_interference(
            lens_grid,
            phase_offset=0.001
        )

        # Phase retrieval
        phase_diff = self.extract_phase(I1, I2)

        # Snake integration
        cumulative = self.integrate_snake(
            phase_diff
        )

        # Reconstruction
        reconstructed = unwrap_phase(
            cumulative
            * self.sensing_wavelength
            / self.science_wavelength
        )

        # Sparse signal
        signal = reconstructed[
            self.valid_subapertures
        ]

        # Full interpolated phase
        full_phase = self.interpolate_surface(
            reconstructed
        )

        full_phase = -np.flip(
            np.rot90(full_phase),
            axis=0
        )

        # Debug
        self.debug = {
            "OPD": OPD.copy(),
            "lens_grid": np.flip(
                np.rot90(lens_grid.copy()),
                axis=0
            ),
            "I1": I1.copy(),
            "I2": I2.copy(),
            "phase_diff": phase_diff.copy(),
            "cumulative": cumulative.copy(),
            "reconstructed": full_phase.copy(),
            "valid_mask": self.valid_subapertures.copy(),
            "signal": signal.copy(),
        }

        # Outputs
        self.signal = np.flipud(signal).flatten()
        self.signal_2D = full_phase
        self.cam = full_phase

        return (
            self.signal,
            self.signal_2D,
            self.cam
        )

    # ============================================================
    # Debug plots
    # ============================================================
    def plot_debug(self):
        if not self.debug:
            raise RuntimeError(
                "Run wfs_measure() first"
            )

        fig, axes = plt.subplots(
            3, 3,
            figsize=(14, 12)
        )
        axes = axes.ravel()

        items = [
            ("Input OPD", self.debug["OPD"]),
            ("Lens Grid", self.debug["lens_grid"]),
            ("Interference I1", self.debug["I1"]),
            ("Interference I2", self.debug["I2"]),
            ("Phase Difference", self.debug["phase_diff"]),
            ("Snake Cumulative", self.debug["cumulative"]),
            ("Reconstructed", self.debug["reconstructed"]),
            ("Valid Mask", self.debug["valid_mask"]),
            ("Signal", self.debug["signal"].reshape(-1, 1)),
        ]

        for ax, (title, img) in zip(axes, items):
            im = ax.imshow(img, origin="lower")
            ax.set_title(title)
            plt.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.04
            )

        plt.tight_layout()
        plt.show()

    # ============================================================
    # OOPAO methods
    # ============================================================
    def __call__(self, OPD, src=None):
        return self.wfs_measure(OPD, src)

    def measure(self, OPD=None, src=None):
        if OPD is None:
            OPD = self.OPD
        return self.wfs_measure(OPD, src)

    def run(self, OPD=None, src=None):
        return self.measure(OPD, src)