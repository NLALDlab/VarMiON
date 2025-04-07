# VarMiON
Python implementation of a Variationally Mimetic Operator Network for time-dependent PDEs

We want to extend the VarMiON, to a time-dependent heat equation defined as
$$C(x){\frac{\partial}{\partial t} u( x,t)}-  \dvg(\beta(\vc x)\nabla u(x, t))= f( x,t)  (x, t) \in \Omega \times [0, \tau],\\
    \beta( x)\nabla u(x, t) \cdot \vc n = \eta(x,t)  ( x, t) \in \Gamma_N \times [0, \tau],\\
    u( x,t)= g( x, t)  ( x, t) \in \Gamma_D\times [0, \tau],\\
    u( x,0)= u_0( x)  (x, t) \in \Omega \times \{0\},
$$
for the temperature field $u: \Omega \times [0,\tau] \rightarrow \mathbb R$ where $u \in L^2([0,\tau]; H^1_{D}(\Omega))$. For the sake of simplicity we consider, from now on, the homogeneous Dirichlet boundary conditions.


Patel, D., Ray, D., Abdelmalik, M.R., Hughes, T.J., Oberai, A.A.: Variationally mimetic
operator networks. *Computer Methods in Applied Mechanics and Engineering* **419** (2024).
https://doi.org/10.1016/j.cma.2023.116536
