# The VarMiON tutorial
This is the source code for the VarMioN tutorial. If you have any comments, corrections or questions, please submit an issue in the issue tracker.



## PDE structure
Python implementation of a Variationally Mimetic Operator Network for time-dependent heat equation with Robin's boundary condition.

The time-dependent heat equation is defined as follow

$$C(x){\frac{\partial}{\partial t} u( x,t)}-  div(\theta( x)\nabla u(x, t))= f( x,t) \quad (x, t) \in \Omega \times [0, \tau],$$

$$\beta( x)\nabla u(x, t) \cdot  n = h(u(x,t)-g(x,t)) \quad ( x, t) \in \Gamma_N \times [0, \tau],$$
 
$$ u( x,0)= u_0( x)  \quad  (x, t) \in \Omega \times \{ 0 \},$$

for the temperature field $u: \Omega \times [0,\tau] \rightarrow \mathbb R$ where $u \in L^2([0,\tau]; H^1(\Omega))$. 

## Generation of PDE data
Here you can find the files to generate the PDE data to train your VarMiON in

* data_generation_heat_equation_robin_2d_fenicsx.py : you can generate and save the data by exploiting the numerical solution of the pde with the Python's Library "FeniCSx";
* data_generation_template.py : you can save youtr data in the correct format.



## References

Patel, D., Ray, D., Abdelmalik, M.R., Hughes, T.J., Oberai, A.A.: Variationally mimetic
operator networks. *Computer Methods in Applied Mechanics and Engineering* **419** (2024).
https://doi.org/10.1016/j.cma.2023.116536


L. Rinaldi, E. Chinellato, P. Martin, and F. Marcuzzi: Exploiting scientific machine learning on embedded
digital twins. *Springer series: Lectures Notes in Computational Science and Engineering - Math to Product*. Submitted 2024.


**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: NLALDlab**
