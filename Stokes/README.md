# The VarMiON tutorial
This is the source code for the VarMiON tutorial. If you have any comments, corrections or questions, please submit an issue in the issue tracker.



## PDE structure
Python implementation of a Variationally Mimetic Operator Network for the time-dependent Stookes equation.

The time-dependent stokes equation is defined as follow:



for the velocity field $u: \Omega \times [0,\tau] \rightarrow \mathbb R$ where $u \in L^2([0,\tau]; H^1(\Omega))$ and pressure field. 

## Generation of PDE data
Here you can find the files to generate the PDE data to train your VarMiON in

* data_generation_ : you can generate and save the data by exploiting the numerical solution of the pde with the Python's Library "FEniCSx"; this file requires version 0.9.0 of DOLFINx, you can run a Docker image with DOLFINx with the command `docker run -ti dolfinx/dolfinx:v0.9.0`
* data_generation_template.py : if you want to use your data, this file shows how to save it in the correct format.

## Requirements

To run this project, create a Conda environment with the required packages. Replace `environment_name` with a name of your choice:

`conda create -n environment_name python=3.11 matplotlib scipy seaborn pytorch pytorch-cuda=11.8 -c pytorch -c nvidia`

Note: Python and PyTorch must be compatible with the CUDA version installed on your system. For more details, see the official PyTorch installation guide https://pytorch.org/get-started/locally/.

To use the environment in Jupyter notebooks:

`source activate base`  
`conda activate environment_name`  
`conda install ipykernel`  
`python -m ipykernel install --user --name environment_name --display-name "Python (environment_name)"`  


## References

Patel, D., Ray, D., Abdelmalik, M.R., Hughes, T.J., Oberai, A.A.: Variationally mimetic
operator networks. *Computer Methods in Applied Mechanics and Engineering* **419** (2024).
https://doi.org/10.1016/j.cma.2023.116536


E. Chinellato, P. Martin, L. Rinaldi, and F. Marcuzzi: Exploiting scientific machine learning on embedded
digital twins. *Springer series: Lectures Notes in Computational Science and Engineering - Math to Product* (2025).
https://link.springer.com/book/9783031957086.

L.Rinaldi, G.G. Giusteri: Variationally mimetic operator network approach to transient viscous flows. 
*In preparation* (2026)


**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: NLALDlab, Marco Dell'Orto, Laura Rinaldi, Enrico Caregnato, Fabio Marcuzzi**
