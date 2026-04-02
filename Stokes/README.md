# The VarMiON tutorial
This is the source code for the VarMiON tutorial. If you have any comments, corrections or questions, please submit an issue in the issue tracker.



## PDE structure
Python implementation of a Variationally Mimetic Operator Network for the time-dependent Stookes equation.

The planar Navier--Stokes problem that describes the behavior of Newtonian viscous fluids in a domain $$\Omega\subseteq R^2$$ with Lipschitz boundary and on a time interval $[0,\tau]$ consists in a system of equations for the velocity $$\mathbf u= (u_1, u_2)^\top $$ and pressure $p$. In the incompressible case, with constant and uniform mass density $\rho>0$, it reads

$$ \rho\left( \frac{\partial \mathbf u}{\partial t} + (\mathbf u \cdot \nabla) \mathbf u\right) = \nabla \cdot \mathbf\sigma(\mathbf u, p) + \mathbf f,$$

$$\nabla \cdot \mathbf u =0,$$

where $$\mathbf f=(f_1, f_2)^\top$$ is the body force per unit volume and $$\sigma(\mathbf u, p)$$ denotes the stress tensor which, for a Newtonian fluid, is given by

$$ \sigma(\mathbf u, p) = 2 \mu \dot{\varepsilon}(\mathbf u ) -p \mathbf I,$$

with $\mathbf I$ the identity tensor, $\mu>0$ the dynamic viscosity, and $\dot{\varepsilon}(\mathbf u)$ the strain-rate tensor defined as

$$ \dot{\varepsilon}(\mathbf u) := \frac{1}{2}\left ( \nabla\mathbf u + (\nabla \mathbf u)^{\top} \right).$$

When inertial forces, that increase with the magnitude of $\mathbf u$, are small compared to viscous forces, the Navier--Stokes equation can be linearized to give, in the time interval $[0,\tau]$, the time-dependent Stokes problem on $\Omega\times [0,\tau]$ as

$$ \rho \frac{\partial \mathbf u}{\partial t}  = -\nabla p+\mu\Delta \mathbf u + \mathbf f,$$

$$\nabla\cdot\mathbf u=0,$$

in which we substituted the Newtonian form of the stress tensor.
The differential system~\eqref{eq:stokes} must be accompanied by initial conditions on 

$$\Omega \times \{0\},$$
given by

$$\mathbf u(\mathbf x, 0) = \mathbf u_0(\mathbf x),\qquad    p(\mathbf x, 0) = p_0(\mathbf x)$$

and boundary conditions, that will be specified for each of the problems solved and for now we summarize in a vector field $\mathbf g$ defined on $\partial\Omega\times[0,\tau]$.



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

## Folder strucrutre

For each example (Cavity flow, flow past a cylinder and contraction flow) you can find:
- the Python script for data generation
- the Python notebook for traning and result comparison
  
Notice that such codes are developed starting from the ones for the Heat problem.

## References

Patel, D., Ray, D., Abdelmalik, M.R., Hughes, T.J., Oberai, A.A.: Variationally mimetic
operator networks. *Computer Methods in Applied Mechanics and Engineering* **419** (2024).
https://doi.org/10.1016/j.cma.2023.116536


E. Chinellato, P. Martin, L. Rinaldi, and F. Marcuzzi: Exploiting scientific machine learning on embedded
digital twins. *Springer series: Lectures Notes in Computational Science and Engineering - Math to Product* (2025).
https://link.springer.com/book/9783031957086.

L.Rinaldi, G.G. Giusteri: Variationally mimetic operator network approach to transient viscous flows. 
*submitted* (2026)


**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Laura Rinaldi, Giulio Giuseppe Giusteri**
