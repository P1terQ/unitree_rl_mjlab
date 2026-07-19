# Installation Guide

## System Requirements

- **Operating System**: Recommended Ubuntu 22.04 
- **GPU**: Nvidia GPU  
- **Driver Version**: Recommended version 550 or later  

---

## 1. Creating a Virtual Environment

It is recommended to run training or deployment programs in a virtual environment. Conda is recommended for creating virtual environments. If Conda is already installed on your system, you can skip step 1.1.

### 1.1 Download and Install MiniConda

MiniConda is a lightweight distribution of Conda, suitable for creating and managing virtual environments. Use the following commands to download and install:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

After installation, initialize Conda:

```bash
~/miniconda3/bin/conda init --all
source ~/.bashrc
```

### 1.2 Create a New Environment

Use the following command to create a virtual environment:

```bash
conda create -n unitree_rl_mjlab python=3.11
```

### 1.3 Activate the Virtual Environment

```bash
conda activate unitree_rl_mjlab
```

---

## 2. Installing

### 2.1 Download the Project

Clone the repository using Git:

```bash
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git
```

### 2.2 Install Dependencies

```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

### 2.3 Install PyTorch

If `nvidia-smi` reports `CUDA Version: 12.8`, install the CUDA 12.8 PyTorch
wheels first to avoid an NVIDIA driver and PyTorch CUDA runtime mismatch:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Verify that CUDA is available after installation:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
PY
```

The expected output should include `torch cuda 12.8` and `cuda available True`.

### 2.4 Install the Project

All other dependencies are specified in the setup.py file.
Navigate to the project root directory and install them with:

```bash
cd unitree_rl_mjlab
pip install -e .
```

If you hit an `AttributeError` mentioning `mjENBL_MULTICCD`, `mujoco` and
`mujoco-warp` are installed at incompatible versions. Reinstall the project
dependencies so `mujoco==3.5.0` matches `mujoco-warp==3.5.0`:

```bash
pip install -e . --upgrade --force-reinstall
```

## Summary

After completing the above steps, you are ready to run the related programs in the virtual environment. If you encounter any issues, refer to the official documentation of each component or check if the dependencies are installed correctly.
