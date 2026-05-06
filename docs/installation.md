# Installation Guideline

This guide covers the recommended installation procedure for SAOS, targeting a Linux environment (e.g., Ubuntu 24). 
Since SAOS relies heavily on multi-threading for performance, the use of **free-threaded Python** (e.g., Python 3.13.2t) is highly recommended.

---

## 1. Prerequisites (Ubuntu)

Install the necessary system dependencies required to build Python and other scientific packages:

```bash
sudo apt update
sudo apt install -y build-essential curl git libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

## 2. Python Environment Setup

We recommend using `pyenv` to manage your Python versions and virtual environments.

### Install `pyenv` and `pyenv-virtualenv`

Follow the official [pyenv installation instructions](https://github.com/pyenv/pyenv) or run the installer:

```bash
curl https://pyenv.run | bash
```
Make sure to add `pyenv` to your `~/.bashrc` or `~/.profile` as instructed by the installer.

### Install Python 3.13 (Free-Threaded)

Install the free-threaded version of Python 3.13 (or newer) and create a virtual environment:

```bash
pyenv install 3.13.2t
pyenv virtualenv 3.13.2t venv_saos
pyenv activate venv_saos
```

To optimize the Python execution, ensure you upgrade `pip` and disable the limited API for CMake packages:

```bash
pip install --upgrade pip
export CMAKE_ARGS="-DPYTHON3_LIMITED_API=OFF"
```

> **Note:** You can add `export PYTHON_GIL=0` to your `~/.bashrc` to ensure the Global Interpreter Lock is disabled when running the free-threaded version.

---

## 3. Clone and Install SAOS

Clone the SAOS repository and install it in editable mode (`-e`) so you can modify the source code easily.

```bash
mkdir -p ~/projects
cd ~/projects

# Clone the repository
git clone https://github.com/nrodlin/SAOS.git
cd SAOS

# Checkout to the development branch (if required)
git checkout develop

# Install SAOS
pip install -e .
```

## 4. Additional Dependencies

Depending on your simulation needs, you may want to install the following dependencies:

```bash
# Required for telemetry streaming
pip install zmq

# Required for handling large data formats (like M7 discrete models)
pip install h5py
```

### Compiling HDF5 and h5py from source (Optional)

If you encounter issues with pre-compiled `h5py` binaries, you can compile HDF5 and `h5py` from source:

```bash
# 1. Compile HDF5
git clone https://github.com/HDFGroup/hdf5.git
cd hdf5
git checkout hdf5-1_14_3
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/hdf5-install -DHDF5_ENABLE_THREADSAFE=ON -DHDF5_BUILD_CPP_LIB=OFF -DBUILD_SHARED_LIBS=ON
make -j$(nproc)
make install
cd ../..

# 2. Compile h5py against the custom HDF5
git clone https://github.com/h5py/h5py.git
cd h5py
export HDF5_DIR=$HOME/hdf5-install
pip install .
```

---

You are now ready to run your first simulation! Head over to the [Tutorials](tutorials/01_intro_to_saos.ipynb) section to learn the basics of SAOS.
