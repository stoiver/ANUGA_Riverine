# ANUGA_Riverine
This repository will serve towards customised ANUGA application (in parallel mode through openmpi) in a riverine system
The required python libraries are listed under anuga_python_libs.yml to accomplish the script for ANUGA model run for hydrodynamic simulation 'MTI_Modeling.py'
The inputs such as terrain file as .tif, boundary line files .shp, unstructured mesh file as .shp,  are kept in directory named "data" inside working directory
The data for Inlet operator (Discharge) and downstream boundary condition (time-varying water level) are in directory named "model_inputs" inside inside working directory

Mesher: UnStructuredMesh (USM) Generator
ANUGA model is paired with a mesh generation package "Mesher" <https://github.com/Chrismarsh/mesher>.
#Installation of Mesher package in wsl
sudo apt-get update
sudo apt-get install -y build-essential  # nice-to-have for compiling things

#if conda isn't set up in WSL yet, install Miniconda first
sudo apt update && sudo apt upgrade -y
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh
###Reload your shell so conda works###
source ~/.bashrc

#add conda-forge & use strict priority (once)
conda config --add channels conda-forge
conda config --set channel_priority strict

#create env and activate
conda create -n mesher_env python=3.11 -y
conda activate mesher_env

#install Mesher (pulls GDAL/VTK etc.)
conda install mesher -y

#Use of Mesher for USM generation
Generated mesh is utilized inside 'anuga_Domain' function utilizing its points (lat/lon) and tri nodes to build domain and interpolate the DEM elevation to the triangles.

#Run ANUGA in parallel mode
export HEADLESS=1
export ANUGA_PARALLEL=openmpi
export OMP_NUM_THREADS=1 
mpirun -n 4 python MTI_Modeling.py
