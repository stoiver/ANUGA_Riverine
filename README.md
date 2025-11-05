# ANUGA_Riverine
This repository will serve towards customised ANUGA application (in parallel mode) in a riverine system
The required python libraries are listed under anuga_install.yml to accomplish the ANUGA model run for hydrodynamic simulation using complete program in MTI_Modeling.py
The inputs such as terrain file as .tif, boundary line files .shp, unstructured mesh file as .shp,  are kept in directory named "data" inside 1_HydrodynamicModeling_ANUGA
The data for Inlet operator (Discharge) and downstream boundary condition (time-varying water level) are in directory named "model_inputs" inside 1_HydrodynamicModeling_ANUGA

ANUGA model simulation paired with a mesh generation package "Mesher" <https://github.com/Chrismarsh/mesher>
Generated mesh is utilized with its nodes to interpolate the 1m DEM (used herein).
Run ANUGA in parallel mode
export HEADLESS=1
export ANUGA_PARALLEL=openmpi
export OMP_NUM_THREADS=1 
mpirun -n 4 python MTI_Modeling.py
