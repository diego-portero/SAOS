# SAOS: Solar Adaptive Optics Simulator

Welcome to the documentation for **SAOS**, a highly-parallelized, object-oriented simulation framework designed specifically for Solar Adaptive Optics.

SAOS is built upon the foundational concepts of OOPAO (Object-Oriented Python Adaptive Optics) but introduces a completely new architecture centered around the `LightPath` paradigm. This shift enables powerful multi-core processing, making it possible to run computationally expensive wide-field Solar AO simulations using free-threaded Python.

---

## 🚀 Key Features

- **Multi-threaded Architecture**: Fully compatible with Python 3.13 free-threaded builds (`-t`), scaling cleanly across multiple CPU cores.
- **Light Path Abstraction**: Allows easy correlation of extended sources (like the Sun) through multiple turbulent layers and deformable mirrors.
- **OOP Design**: Simple to configure and extend, with individual classes for `Telescope`, `Atmosphere`, `DeformableMirror`, `ShackHartmann`, etc.
- **Python Only**: No MATLAB dependencies. All heavy lifting is handled by Numpy/Scipy, with optional PyTorch support.

---

## 📖 Where to go next?

- **[Installation](installation.md)**: Step-by-step instructions on setting up your environment for maximum performance.
- **[Tutorials](tutorials/01_intro_to_saos.ipynb)**: Pedagogical notebooks explaining how to build and run your first simulations.
- **[API Reference](api/atmosphere.md)**: Detailed documentation of the internal classes and methods.
