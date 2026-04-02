# SPDX-License-Identifier: GPL-3.0
# Copyright (c) 2025 NLALDlab
# Authors: NLALDlab


import os
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime

# works with dolfinx 0.9.0


import dolfinx 
from dolfinx import mesh
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
from time import time


from dolfinx import fem, cpp
from dolfinx.mesh import locate_entities_boundary, meshtags
from ufl import ds
from petsc4py import PETSc

from dolfinx import fem
from ufl import VectorElement, FiniteElement, MixedElement
import ufl

from dolfinx.fem import Function, FunctionSpace, dirichletbc, locate_dofs_topological
from dolfinx import plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
import scipy.sparse
                  
import gmsh
from dolfinx.io import gmshio

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np


from dolfinx.fem import Constant, Function, functionspace, assemble_scalar, dirichletbc, form, locate_dofs_geometrical
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from dolfinx.io import VTXWriter
from dolfinx.mesh import create_unit_square
from dolfinx.plot import vtk_mesh
from basix.ufl import element
from ufl import (FacetNormal, Identity, TestFunction, TrialFunction,
                 div, dot, ds, dx, inner, lhs, nabla_grad, rhs, sym)


from dolfinx.io import *
from dolfinx.io import XDMFFile
from dolfinx.geometry import *


class PDE(ABC):
    def __init__(self, mesh: dolfinx.mesh.Mesh, V: dolfinx.fem.FunctionSpace, Q: dolfinx.fem.FunctionSpace) -> None:
        self.fem_solver = ...
        self.solution = None
        self.mesh = mesh
        self.V = V
        self.Q = Q

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def generate_parameters(self):
        pass

    @abstractmethod
    def solve(self):
        pass


    
                  
class pde_heat_eq__robin(PDE):
    def __init__(self, mesh, V,Q, times_train):
        super().__init__(mesh=mesh, V=V, Q=Q)
 
        self.t = times_train[0]
        self.T = times_train[-1]
        self.n_times = len(times_train)
        self.dt = times_train[1] - times_train[0] 

        self.num_points = len(self.mesh.geometry.x)
        self.tdim = self.mesh.topology.dim
        self.fdim = self.tdim - 1
                          
        
        self.mu=0
        self.ff=0
        self.f_vec = [dolfinx.fem.Function(self.V) for _ in range(self.n_times)]
        self.solution = [dolfinx.fem.Function(self.V) for _ in range(self.n_times)]
        self.solution_pressure = [dolfinx.fem.Function(self.Q) for _ in range(self.n_times)]
      
        
    def __call__(self):
        self.generate_parameters()
        self.solve()
        return self.mu, self.f_vec,  self.solution, self.solution_pressure 


    def generate_parameters(self):
        
        self.mu =  np.random.uniform(0.1, 0.9)
        self.ff = np.random.uniform(0.02, 0.9)
        self.solution[0].x.array[:] = np.zeros(np.shape(self.solution[0].x.array[:]))
        self.solution_pressure[0].x.array[:] = np.zeros(self.num_points)
    
    def solve(self):
        
        # Testing evaluation on P
        from dolfinx.geometry import bb_tree, BoundingBoxTree, compute_collisions_points, compute_colliding_cells

        P = [0.01,0.2, 0.0] 
        tree = bb_tree(self.mesh, self.mesh.topology.dim)

        cell_candidates = compute_collisions_points(tree, np.array(P))
        cell = compute_colliding_cells(self.mesh, cell_candidates, P)    
        
        
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        p = TrialFunction(self.Q)
        q = TestFunction(self.Q)
       
        #boundary conditions
        tol=1.e-6
        def walls(x):
            return np.logical_or(x[1]< 0+ tol, x[0]< 0+ tol, x[0]> 1- tol)
        
        wall_dofs = locate_dofs_geometrical(self.V, walls)
        u_noslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
        bc_noslip = dirichletbc(u_noslip, wall_dofs, self.V)

        V1 = fem.VectorFunctionSpace(self.mesh, ("Lagrange", 1))
        class InletVelocity:
            def __init__(self, g, t):
                self.g = g
                self.t = t   

            def __call__(self, x):
                values = np.zeros((mesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType)
                values[0] = self.g  *8* (1 + np.tanh(8 * (self.t - 0.5))) * x[0]**2 * (1 - x[0])**2
                values[1] =0
                return values
        
        

        # Inlet
        u_inlet = Function(self.V)
        inlet_velocity = InletVelocity(self.ff ,self.t) 
        u_inlet.interpolate(inlet_velocity)
        
        inlet_dofs = locate_dofs_geometrical(self.V, lambda x: x[1] > 1-tol)  # Definisci inlet_x_threshold
        bcu_inlet = dirichletbc(u_inlet, inlet_dofs)  
        bcu = [bc_noslip, bcu_inlet]

        
        def inflow(x):
            return np.isclose(x[0], 0)
        
        inflow_dofs = locate_dofs_geometrical(self.Q, inflow)
        bc_inflow = dirichletbc(PETSc.ScalarType(0), inflow_dofs, self.Q)

        def outflow(x):
            return np.isclose(x[0], 1)
        
        
        outflow_dofs = locate_dofs_geometrical(self.Q, outflow)
        bc_outflow = dirichletbc(PETSc.ScalarType(0), outflow_dofs, self.Q)
        
        

        def top(x):
            return np.isclose(x[1], 0)
        
        
        top_dofs = locate_dofs_geometrical(self.Q, top)
        bc_top = dirichletbc(PETSc.ScalarType(0), top_dofs, self.Q)
       
        
      
        bcp = [bc_top]
        
        u_n = Function(self.V)
        u_n.name = "u_n"
        u_n.x.array[:] = self.solution[0].x.array[:] 
        
        U = 0.5 * (u_n + u)
        n = FacetNormal(mesh)
        f = Constant(mesh, PETSc.ScalarType((0,0,0)))
        k = Constant(mesh, PETSc.ScalarType(self.dt))
        mu = Constant(mesh, PETSc.ScalarType(1/self.mu))
        rho = Constant(mesh, PETSc.ScalarType(1))
        
        
        # Define strain-rate tensor
        def epsilon(u):
            return sym(nabla_grad(u))

        # Define stress tensor
        def sigma(u, p):
            return 2 * mu * epsilon(u) - p * Identity(len(u))


        # Define the variational problem for the first step
        p_n = Function(self.Q)
        p_n.name = "p_n"
        p_n.x.array[:] = self.solution_pressure[0].x.array[:] 
            
        F1 = rho * dot((u - u_n) / k, v) * dx
        # F1 += rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx
        F1 += inner(sigma(U, p_n), epsilon(v)) * dx
        F1 += dot(p_n * n, v) * ds - dot(mu * nabla_grad(U) * n, v) * ds
        F1 -= dot(f, v) * dx
        a1 = form(lhs(F1))
        L1 = form(rhs(F1))
        
        A1 = assemble_matrix(a1, bcs=bcu)
        A1.assemble()
        b1 = create_vector(L1)
        
        
        # Define variational problem for step 2
        u_ = Function(self.V)
        a2 = form(dot(nabla_grad(p), nabla_grad(q)) * dx)
        L2 = form(dot(nabla_grad(p_n), nabla_grad(q)) * dx - (rho / k) * div(u_) * q * dx)
        A2 = assemble_matrix(a2, bcs=bcp)
        A2.assemble()
        b2 = create_vector(L2)

        # Define variational problem for step 3
        p_ = Function(self.Q)
        a3 = form(rho * dot(u, v) * dx)
        L3 = form(rho * dot(u_, v) * dx - k * dot(nabla_grad(p_ - p_n), v) * dx)
        A3 = assemble_matrix(a3)
        A3.assemble()
        b3 = create_vector(L3)
        
        
        # Solver for step 1
        solver1 = PETSc.KSP().create(mesh.comm)
        solver1.setOperators(A1)
        solver1.setType(PETSc.KSP.Type.BCGS)
        pc1 = solver1.getPC()
        pc1.setType(PETSc.PC.Type.HYPRE)
        pc1.setHYPREType("boomeramg")

        # Solver for step 2
        solver2 = PETSc.KSP().create(mesh.comm)
        solver2.setOperators(A2)
        solver2.setType(PETSc.KSP.Type.BCGS)
        pc2 = solver2.getPC()
        pc2.setType(PETSc.PC.Type.HYPRE)
        pc2.setHYPREType("boomeramg")

        # Solver for step 3
        solver3 = PETSc.KSP().create(mesh.comm)
        solver3.setOperators(A3)
        solver3.setType(PETSc.KSP.Type.CG)
        pc3 = solver3.getPC()
        pc3.setType(PETSc.PC.Type.SOR)
        
        
        self.f_vec[0].x.array[:] = np.zeros(np.shape(self.solution[0].x.array[:]))
        self.t=0 
        
        for i in range(self.n_times-1):
        # Update current time step
            
            self.t += self.dt
            inlet_velocity = InletVelocity( self.ff ,self.t)  
            u_inlet.interpolate(inlet_velocity)
            bcu_inlet = dirichletbc(u_inlet, inlet_dofs)
            bcu = [bc_noslip, bcu_inlet]

            
            # Step 1: Tentative veolcity step
            with b1.localForm() as loc_1:
                loc_1.set(0)
            assemble_vector(b1, L1)

            A1 = assemble_matrix(a1, bcs=bcu)
            A1.assemble()

            
            apply_lifting(b1, [a1], [bcu])
            b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b1, bcu)
            solver1.solve(b1, u_.vector)
            u_.x.scatter_forward()

            # Step 2: Pressure corrrection step
            with b2.localForm() as loc_2:
                loc_2.set(0)
            assemble_vector(b2, L2)
            apply_lifting(b2, [a2], [bcp])
            b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b2, bcp)
            solver2.solve(b2, p_.vector)
            p_.x.scatter_forward()

            # Step 3: Velocity correction step
            with b3.localForm() as loc_3:
                loc_3.set(0)
            assemble_vector(b3, L3)
            b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            solver3.solve(b3, u_.vector)
            u_.x.scatter_forward()
            
            # Update variable with solution form this time step
            u_n.x.array[:] = u_.x.array[:]
            p_n.x.array[:] = p_.x.array[:]

            self.solution[i+1].x.array[:] = u_n.x.array  
        
            self.solution_pressure[i+1].x.array[:] = p_n.x.array 
            self.f_vec[i+1].x.array[:] = u_inlet.x.array[:]

        b1.destroy()
        b2.destroy()
        b3.destroy()
        solver1.destroy()
        solver2.destroy()
        solver3.destroy()
        

                                    

def generate_data(num_pdes, mesh, V, Q, param_pts, param_bry_pts, temp_pts, times, times_train, nb=0):
        
        # num_pdes int
        # param_pts nparray with shape (n_param_pts, 3)
        # temp_pts nparray with shape (n_temp_pts, 3), only spatial grid
        # nb = 0: there are only grid points
        # nb > 0: there are nb boundary points after the grid points

        n_param_pts = len(param_pts)
        n_param_bry_pts = len(param_bry_pts)
        n_temp_pts = len(temp_pts)

        # Reorder 'lexicographically' temp_pts
        if nb==0:
            indices = np.lexsort(temp_pts.T)
            temp_pts = temp_pts[indices]
        else:
            indices = np.lexsort(temp_pts[:-nb].T)
            temp_pts[:-nb] = temp_pts[:-nb][indices]
        #endif
            
        #print(n_param_pts, n_param_bry_pts ,n_temp_pts)
        # Find cells on which to evaluate param_pts and save them in cells_param_pts
        bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
        potential_colliding_cells = dolfinx.geometry.compute_collisions_points(bb_tree, param_pts)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, potential_colliding_cells, param_pts)

        cells_param_pts = []
        for i in range(len(param_pts)):
                
                cells_param_pts.append(colliding_cells.links(i)[0])
        #endfor
                  
                  
        # Find cells on which to evaluate param_bry_pts and save them in cells_param_bry_pts
        bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
        potential_colliding_cells = dolfinx.geometry.compute_collisions_points(bb_tree, param_bry_pts)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, potential_colliding_cells, param_bry_pts)

        cells_param_bry_pts = []
        for i in range(len(param_bry_pts)):
                cells_param_bry_pts.append(colliding_cells.links(i)[0])
        #endfor             
    
                  
            
        # Find cells on which to evaluate temp_pts and save them in cells_temp_pts
        bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
        potential_colliding_cells = dolfinx.geometry.compute_collisions_points(bb_tree, temp_pts)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, potential_colliding_cells, temp_pts)

        cells_temp_pts = []
        for i in range(len(temp_pts)):
                cells_temp_pts.append(colliding_cells.links(i)[0])
        #endfor
            

        pde = pde_heat_eq__robin(mesh, V,Q, times_train)

        data = dict(
            mu = np.zeros(shape = num_pdes, dtype = np.float32),
            f_vec= np.zeros(shape = (num_pdes, len(times),n_param_pts*3), dtype = np.float32),
            solution = np.zeros(shape = (num_pdes, len(times),n_param_pts*3), dtype = np.float32),
            solution_pressure= np.zeros(shape = (num_pdes, len(times),n_param_pts), dtype = np.float32),
        )
        
        dir_name = lambda t: f"elasticity_2d_{num_pdes}/{t}"
        dir_record = lambda t: f"elasticity_2d_{num_pdes}/record_{t}.npy"

        os.system(f'rm -r {dir_name("")}')
        os.system(f'mkdir {dir_name("")}')
        # save mesh (not useful now)
        with dolfinx.io.XDMFFile(pde.mesh.comm, filename=dir_name("/mesh.xdmf"), file_mode="w") as file:
            file.write_mesh(pde.mesh)

        # save points and times 
        with open(dir_name("eval_pts"), 'wb') as file:
            np.save(file, param_pts[:,:pde.tdim])
            np.save(file, param_bry_pts[:,:pde.tdim])
            np.save(file, temp_pts[:,:pde.tdim])
            np.save(file, times)
            np.save(file, times_train)
            np.save(file, nb)
            
        
        # save evaluations
        for i in range(num_pdes):

            
            if i%10==0: print("generating pde #", i)      
            pde()
            data["mu"][i] = pde.mu
                  
            for j in range(len(times)):
                data["f_vec"][i,j, :] = pde.f_vec[j * int(len(times_train)/len(times))].eval(x = temp_pts, cells = cells_temp_pts).flatten()
                data["solution"][i,j, :] = pde.solution[j * int(len(times_train)/len(times))].eval(x = temp_pts, cells = cells_temp_pts).flatten()
                data["solution_pressure"][i,j, :] = pde.solution_pressure[j * int(len(times_train)/len(times))].eval(x = temp_pts, cells = cells_temp_pts).flatten()


             
            with open(dir_record(i), 'wb') as file:                  
                np.save(file, data["mu"][i])
                np.save(file, data["f_vec"][i,:,:])
                np.save(file, data["solution"][i, :,:]) 
                np.save(file, data["solution_pressure"][i, :,:])
                  

                  
        #endfor
        

if __name__ == '__main__':
                                    
    # add or not points on the boundary
    boundary = False

    # number of pde instances to solve             
    num_pdes = 1000

                  
    # dolfinx mesh used to solve the pdes
    R = 0.0#0.5    # half of distance between plates
    L = 1.0#2.0    # length

                  
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("rettangolo_con_buco")

    # Rectangle
    rettangolo_grande = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 1.0, 1.0)

    gmsh.model.occ.synchronize()

    # Surface
    superficie_finale = rettangolo_grande 

    # Groups
    gmsh.model.addPhysicalGroup(2, [superficie_finale])
    gmsh.model.setPhysicalName(2, 1, "Dominio_senza_disco")

    # Mesh resolution
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.04)

    # Mesh options
    gmsh.option.setNumber("Mesh.Algorithm", 6)        # Delaunay triangulation
    gmsh.option.setNumber("Mesh.ElementOrder", 1)     # 1st order Elements
    gmsh.option.setNumber("Mesh.RecombineAll", 0)     # Triangles

    # 2D mesh 
    gmsh.model.mesh.generate(2)

    # Conversion to Dolfinx
    meshbr, cell_markers_b, facet_markers_b = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0)

    # Gmsh
    gmsh.finalize()              

    # Names
    cell_markers_b.name = f"{meshbr.name}_cells"
    facet_markers_b.name = f"{meshbr.name}_facets"

    # Mesh
    mesh = meshbr
    tdim = mesh.topology.dim
                  
  
    print("Mesh dimension:", mesh.geometry.dim)
    print("Topology dimension:", mesh.topology.dim)
    print("Cell name:", mesh.topology.cell_name())
    from ufl import VectorElement
    
    # Elements definition
    v_cg2 =  VectorElement("Lagrange", mesh.topology.cell_name(), 2, dim=mesh.geometry.dim)
    s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)

    
    # Function spaces
    V = fem.VectorFunctionSpace(mesh, ("Lagrange", 2))
    Q = fem.FunctionSpace(mesh, ("Lagrange", 1))    

                  
    #### Define points and times at which to evaluate the parameters and the solution ####

    # define evaluation points for parameters and temperature
    # param_points: nparray (n_param_points, 3) on which we sample the parameters (just a grid for now)
    # temp_points: nparray (n_temp_points, 3) on which we sample the temperature (a grid + boundary points if boundary==True)



    times = np.linspace(0,2,20)  
    times_train = np.linspace(0,2,2000)
    
    # grid
    n = 15
    v = np.linspace(0,1,n)
    grid_points0 = np.array([[v[i], v[j], 0] for j in range(n) for i in range(n)])
    grid_points = grid_points0        
                  
    # param boundary points
    m = 10
    w = np.linspace(0,1,m)

    bottom = np.column_stack((w, np.zeros(m)))
    top = np.column_stack((w, np.ones(m)))
    left = np.column_stack((np.zeros(m-2), w[1:-1]))
    right = np.column_stack((np.ones(m-2), w[1:-1]))

    param_bry_points = np.vstack((bottom, top, left, right))
    param_bry_points = np.column_stack((param_bry_points, np.zeros(param_bry_points.shape[0])))
          
                  

    # boundary points
    m = 57
    w = np.linspace(0,1,m)

    bottom = np.column_stack((w, np.zeros(m)))
    top = np.column_stack((w, np.ones(m)))
    left = np.column_stack((np.zeros(m-2), w[1:-1]))
    right = np.column_stack((np.ones(m-2), w[1:-1]))

    bry_points = np.vstack((bottom, top, left, right))
    bry_points = np.column_stack((bry_points, np.zeros(bry_points.shape[0])))


    # param points
    param_points = grid_points
    print("#param points: ", param_points.shape[0])

    if boundary:
        # stack points
        temp_points = np.vstack((grid_points, bry_points))
        nb = len(bry_points)
    else:
        temp_points = grid_points
        nb = 0
    #endif
    print("#temp points: ", temp_points.shape[0])

    print(f"\n\nData Generation - {datetime.now()} \n\n")
    generate_data(num_pdes, mesh, V,Q, param_points, param_bry_points, temp_points, times,  times_train, nb)
    print(f"\n\nData Generated - {datetime.now()} \n\n")     
