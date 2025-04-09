# VarMiON
Python implementation of a Variationally Mimetic Operator Network for time-dependent PDEs

We want to extend the VarMiON, to a time-dependent heat equation defined as

$$C(x){\frac{\partial}{\partial t} u( x,t)}-  div(\beta( x)\nabla u(x, t))= f( x,t) \quad (x, t) \in \Omega \times [0, \tau],$$

$$\beta( x)\nabla u(x, t) \cdot  n = \eta(x,t) \quad ( x, t) \in \Gamma_N \times [0, \tau],$$

$$u( x,t)= g( x, t)  \quad  ( x, t) \in \Gamma_D\times [0, \tau],$$
 
$$ u( x,0)= u_0( x)  \quad  (x, t) \in \Omega \times \{0\},$$

for the temperature field $u: \Omega \times [0,\tau] \rightarrow \mathbb R$ where $u \in L^2([0,\tau]; H^1_{D}(\Omega))$. 
For the sake of simplicity we consider the homogeneous Dirichlet boundary conditions $g( x, t)=0$.

# References

Patel, D., Ray, D., Abdelmalik, M.R., Hughes, T.J., Oberai, A.A.: Variationally mimetic
operator networks. *Computer Methods in Applied Mechanics and Engineering* **419** (2024).
https://doi.org/10.1016/j.cma.2023.116536


L. Rinaldi, E. Chinellato, P. Martin, and F. Marcuzzi: Exploiting scientific machine learning on embedded
digital twins. *Springer series: Lectures Notes in Computational Science and Engineering - Math to Product*. Submitted 2024.


**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: NLALDlab**
