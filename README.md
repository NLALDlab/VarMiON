# VarMiON
Python implementation of a Variationally Mimetic Operator Network for time-dependent PDEs

We want to extend the VarMiON, to a time-dependent heat equation defined as
\begin{align}\label{eq:prob_heat_eq}
   C(\vc x){\frac{\partial}{\partial t} u(\vc x,t)}-  \dvg(\beta(\vc x)\nabla u(\vc x, t))&= f(\vc x,t)  &(\vc x, t) \in \Omega \times [0, \tau],\\
    \beta(\vc x)\nabla u(\vc x, t) \cdot \vc n &= \eta(\vc x,t)  &(\vc x, t) \in \Gamma_N \times [0, \tau],\\
    u(\vc x,t)&= g(\vc x, t)  &(\vc x, t) \in \Gamma_D\times [0, \tau],\\
    u(\vc x,0)&= u_0(\vc x)  &(\vc x, t) \in \Omega \times \{0\},
\end{align}
for the temperature field $u: \Omega \times [0,\tau] \rightarrow \mathbb R$ where $u \in L^2([0,\tau]; H^1_{D}(\Omega))$. For the sake of simplicity we consider, from now on, the homogeneous Dirichlet boundary conditions.


Patel, D., Ray, D., Abdelmalik, M.R., Hughes, T.J., Oberai, A.A.: Variationally mimetic
operator networks. *Computer Methods in Applied Mechanics and Engineering* **419** (2024).
https://doi.org/10.1016/j.cma.2023.116536
