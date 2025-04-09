# SPDX-License-Identifier: GPL-3.0
# Copyright (c) 2025 NLALDlab
# Authors: NLALDlab


import os
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime

# works with dolfinx 0.9.0
import dolfinx, dolfinx.fem.petsc, petsc4py, ufl
import mpi4py.MPI as MPI



# contains the classes:
# PDE --> gives the structure of a class that solves a PDE
# pde_heat_eq__robin(PDE) --> solves the heat equation with Robin boundary conditions
#M le classi relative ad altre pde possono essere cancellate o usate in seguito...




class PDE(ABC):
    def __init__(self, mesh: dolfinx.mesh.Mesh, function_space: dolfinx.fem.FunctionSpace) -> None:
        self.fem_solver = ...
        self.solution = None
        self.mesh = mesh
        self.function_space = function_space

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def generate_parameters(self):
        pass

    @abstractmethod
    def solve(self):
        pass

#M non serve per robin
class pde_steady_state_heat_eq__dirichlet(PDE):
    def __init__(self, mesh, function_space):
        super().__init__(mesh=mesh, function_space=function_space)
             
        #self.dtype = np.float64#np.float32
            
        self.tdim = mesh.topology.dim
        self.fdim = self.tdim - 1
        
        #self.facets = dolfinx.mesh.locate_entities_boundary(mesh=self.mesh, dim=self.fdim,
        #    marker=lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)))

        #self.facets = dolfinx.mesh.locate_entities_boundary(msh=self.mesh, dim=self.fdim,
        #    marker=lambda x: True)
        
        #self.degrees_of_freedom = dolfinx.fem.locate_dofs_topological(V=self.function_space, entity_dim=self.fdim, entities=self.facets)
        #self.boundary_conditions = dolfinx.fem.dirichletbc(value=petsc4py.PETSc.ScalarType(0),
        #                dofs=self.degrees_of_freedom, V=self.function_space)
        
        
        
        def boundary_D(x):
            #return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1))
            a = np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1))
            b = np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))
            return np.logical_or(a, b)

                
        self.dofs_D = dolfinx.fem.locate_dofs_geometrical(self.function_space, boundary_D)

        self.boundary_conditions = dolfinx.fem.dirichletbc(value=petsc4py.PETSc.ScalarType(0), dofs = self.dofs_D, V=self.function_space)

        
        
        
        length_scale_f = 0.2
        length_scale_theta = 0.4
        distances_matrix = np.sqrt(np.sum(
            (self.mesh.geometry.x[:, 0:self.tdim][:, np.newaxis, :] - self.mesh.geometry.x[:, 0:self.tdim][np.newaxis, :, :])**2, axis=2))
        self.covariance_matrix_f = np.exp(-distances_matrix ** 2 / (2 * length_scale_f ** 2))
        self.covariance_matrix_theta = np.exp(-distances_matrix ** 2 / (2 * length_scale_theta ** 2))

        self.fem_solver = dolfinx.fem.petsc.LinearProblem
        self.fem_solver.petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
        self.f = dolfinx.fem.Function(self.function_space) 
        self.f.name = 'f'
        self.theta = dolfinx.fem.Function(self.function_space)
        self.theta.name = 'theta'
        self.solution = dolfinx.fem.Function(self.function_space)
        self.solution.name = 'solution'
        

    def __call__(self):
        self.generate_parameters()
        self.solve()
        return self.f, self.theta, self.solution

    def generate_parameters(self):
        num_points = len(self.mesh.geometry.x)
        f = np.random.multivariate_normal(mean=np.zeros(num_points),
                                          cov=self.covariance_matrix_f)
        theta = np.random.multivariate_normal(mean=np.zeros(num_points),
                                              cov=self.covariance_matrix_theta)
        self.f.x.array[:] = (f - np.min(f)) / (np.max(f) - np.min(f)) * 0.97 + 0.02
        self.theta.x.array[:] = (theta - np.min(theta)) / (np.max(theta) - np.min(theta)) * 0.97 + 0.02

    def solve(self):
        u = ufl.TrialFunction(self.function_space)
        v = ufl.TestFunction(self.function_space)
        a = self.theta * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(self.f, v) * ufl.dx
        problem = self.fem_solver(a, L, bcs=[self.boundary_conditions])
        self.solution = problem.solve()



    def generate_and_save_fem_dataset(self) -> None:
        os.system(f'rm -r {self.dir("")}')
        os.system(f'mkdir {self.dir("")}')
        with dolfinx.io.XDMFFile(self.pde.mesh.comm, filename=self.dir("/mesh.xdmf"), file_mode="w") as file:
            file.write_mesh(self.pde.mesh)
        for i in range(self.num_pdes):
            if i%10==0: print("generating pde #", i)      
            self.pde()

            #with open(self.dir_record(i), 'wb') as file:
            #    np.save(file, self.pde.f.vector)       
            #    np.save(file, self.pde.theta.vector)
            #    np.save(file, self.pde.solution.vector)

            self.data["f"][i, :] = self.pde.f.eval(x = self.evaluation_points, cells = self.cells_evaluation).flatten()
            self.data["theta"][i, :] = self.pde.theta.eval(x = self.evaluation_points, cells = self.cells_evaluation).flatten()
            self.data["solution"][i, :] = self.pde.solution.eval(x = self.evaluation_points, cells = self.cells_evaluation).flatten()

            with open(self.dir_record(i), 'wb') as file:
                np.save(file, self.data["f"][i,:])       
                np.save(file, self.data["theta"][i,:])
                np.save(file, self.data["solution"][i,:])

class pde_heat_eq__dirichlet(PDE):
    def __init__(self, mesh, function_space, times):
        super().__init__(mesh=mesh, function_space=function_space)
 
        self.t = times[0]
        self.T = times[-1]
        self.n_times = len(times)
        self.num_steps = self.n_times #M: da cambiare
        self.dt = times[1] - times[0]  #self.T/self.num_steps

        self.num_points = len(self.mesh.geometry.x)

        self.tdim = self.mesh.topology.dim
        self.fdim = self.tdim - 1
        
        #self.facets = dolfinx.mesh.locate_entities_boundary(mesh=self.mesh, dim=self.fdim,
        #    marker=lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)))
        
        self.mesh.topology.create_connectivity(self.fdim, self.tdim)
        self.facets = dolfinx.mesh.exterior_facet_indices(self.mesh.topology)


        self.degrees_of_freedom = dolfinx.fem.locate_dofs_topological(V=self.function_space, entity_dim=self.fdim, entities=self.facets)
        self.boundary_conditions = dolfinx.fem.dirichletbc(value=petsc4py.PETSc.ScalarType(0),
                        dofs=self.degrees_of_freedom, V=self.function_space)

        length_scale_f = 0.2
        length_scale_theta = 0.4
        length_scale_c = 0.4
        distances_matrix = np.sqrt(np.sum(
            (self.mesh.geometry.x[:, 0:self.tdim][:, np.newaxis, :] - self.mesh.geometry.x[:, 0:self.tdim][np.newaxis, :, :]) ** 2,
            axis=2))
        self.covariance_matrix_f = np.exp(-distances_matrix ** 2 / (2 * length_scale_f ** 2))
        self.covariance_matrix_theta = np.exp(-distances_matrix ** 2 / (2 * length_scale_theta ** 2))
        self.covariance_matrix_c = np.exp(-distances_matrix ** 2 / (2 * length_scale_c ** 2))

        #self.fem_solver = dolfinx.fem.petsc.LinearProblem
        #self.fem_solver.petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}

        self.f_vec = [dolfinx.fem.Function(self.function_space) for _ in range(self.num_steps)] 
        self.f = dolfinx.fem.Function(self.function_space)
        self.f.name = 'f'
        self.theta = dolfinx.fem.Function(self.function_space)
        self.theta.name = 'theta'
        self.c = dolfinx.fem.Function(self.function_space)
        self.c.name = 'c'
        self.solution_vec = [dolfinx.fem.Function(self.function_space) for _ in range(self.num_steps)]
        #self.solution.name = 'solution'
        

    def __call__(self):
        self.generate_parameters()
        self.solve()
        return self.f_vec, self.theta, self.c, self.solution_vec

    def generate_parameters(self): # generates only spatial parameters theta and c
        num_points = len(self.mesh.geometry.x)
        #f = np.random.multivariate_normal(mean=np.zeros(num_points),
        #                                  cov=self.covariance_matrix_f)
        
        theta = np.random.multivariate_normal(mean=np.zeros(num_points),
                                              cov=self.covariance_matrix_theta)
        c = np.random.multivariate_normal(mean=np.zeros(num_points),
                                              cov=self.covariance_matrix_c)
        
        
        for i in range(1, self.num_steps):
            f = np.random.multivariate_normal(mean=np.zeros(self.num_points), cov=self.covariance_matrix_f)
            self.f_vec[i].x.array[:] = (f - np.min(f)) / (np.max(f) - np.min(f)) * 0.97 + 0.02 

        self.theta.x.array[:] = (theta - np.min(theta)) / (np.max(theta) - np.min(theta)) * 0.97 + 0.02
        self.c.x.array[:] = (c - np.min(c)) / (np.max(c) - np.min(c)) * 0.97 + 0.02

    #def generate_f(self):
    #    f = np.random.multivariate_normal(mean=np.zeros(self.num_points),
    #                                      cov=self.covariance_matrix_f)
    #    self.f.vector[:] = (f - np.min(f)) / (np.max(f) - np.min(f)) * 0.97 + 0.02
    

    def solve(self):
        # setup solver
        u_n = dolfinx.fem.Function(self.function_space)
        u_n.name = "u_n"
        u_n.x.array[:] = np.zeros(self.num_points)
        
        u = ufl.TrialFunction(self.function_space)
        v = ufl.TestFunction(self.function_space)
        
        a = self.c * u * v * ufl.dx + self.dt * self.theta * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = (self.c * u_n + self.dt * self.f) * v * ufl.dx

        bilinear_form = dolfinx.fem.form(a)
        linear_form = dolfinx.fem.form(L)

        A = dolfinx.fem.petsc.assemble_matrix(bilinear_form, bcs=[self.boundary_conditions])
        A.assemble()
        b = dolfinx.fem.petsc.create_vector(linear_form)

        uh = dolfinx.fem.Function(self.function_space)

        #
        self.fem_solver = petsc4py.PETSc.KSP().create(self.mesh.comm)
        self.fem_solver.setOperators(A)
        self.fem_solver.setType(petsc4py.PETSc.KSP.Type.PREONLY)
        self.fem_solver.getPC().setType(petsc4py.PETSc.PC.Type.LU)
        #
        
        # solve
        #t = self.t
        for i in range(1, self.num_steps):
            #t += self.dt

            # Generate f at current time step and save it
            #self.generate_f()
            #self.f_vec[i+1].x.array[:] = self.f.x.array

            self.f.x.array[:] = self.f_vec[i].x.array
            
            # Update the right hand side reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            dolfinx.fem.petsc.assemble_vector(b, linear_form)
        
            # Apply Dirichlet boundary condition to the vector
            dolfinx.fem.petsc.apply_lifting(b, [bilinear_form], [[self.boundary_conditions]])
            b.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD_VALUES, mode=petsc4py.PETSc.ScatterMode.REVERSE)
            dolfinx.fem.petsc.set_bc(b, [self.boundary_conditions])
        
            # Solve linear problem
            self.fem_solver.solve(b, uh.x.petsc_vec)
            uh.x.scatter_forward()
        
            # Update solution at previous time step (u_n)
            u_n.x.array[:] = uh.x.array
            
            # Save solution at current time step
            self.solution_vec[i].x.array[:] = u_n.x.array  
        #endfor
                  
                  
                  
                  
class pde_heat_eq__robin(PDE):
    def __init__(self, mesh, function_space, times):
        super().__init__(mesh=mesh, function_space=function_space)
 
        self.t = times[0]
        self.T = times[-1]
        self.n_times = len(times)
        self.dt = times[1] - times[0]  #self.T/self.num_steps

        self.num_points = len(self.mesh.geometry.x)
        self.tdim = self.mesh.topology.dim
        self.fdim = self.tdim - 1
                          
        length_scale_u0 = 0.2
        length_scale_f = 0.2
        length_scale_theta = 0.4
        length_scale_c = 0.4
        length_scale_g = 0.4
                  
                  
        distances_matrix = np.sqrt(np.sum(
            (self.mesh.geometry.x[:, 0:self.tdim][:, np.newaxis, :] - self.mesh.geometry.x[:, 0:self.tdim][np.newaxis, :, :]) ** 2,
            axis=2))
                  
        self.covariance_matrix_u0 = np.exp(-distances_matrix ** 2 / (2 * length_scale_u0 ** 2))
        self.covariance_matrix_f = np.exp(-distances_matrix ** 2 / (2 * length_scale_f ** 2))
        self.covariance_matrix_theta = np.exp(-distances_matrix ** 2 / (2 * length_scale_theta ** 2))
        self.covariance_matrix_c = np.exp(-distances_matrix ** 2 / (2 * length_scale_c ** 2))
        self.covariance_matrix_g = np.exp(-distances_matrix ** 2 / (2 * length_scale_g ** 2))

                  
        self.f_vec = [dolfinx.fem.Function(self.function_space) for _ in range(self.n_times-1)] 
        self.f = dolfinx.fem.Function(self.function_space)
        self.f.name = 'f'
                  
        self.theta = dolfinx.fem.Function(self.function_space)
        self.theta.name = 'theta'
                  
        self.c = dolfinx.fem.Function(self.function_space)
        self.c.name = 'c'
                  
        self.g_vec = [dolfinx.fem.Function(self.function_space) for _ in range(self.n_times-1)]
        self.g = dolfinx.fem.Function(self.function_space)
        self.g.name = 'f'  
                  
        self.solution_vec = [dolfinx.fem.Function(self.function_space) for _ in range(self.n_times)]
        
    def __call__(self):
        self.generate_parameters()
        self.solve()
        return self.f_vec, self.theta, self.c, self.g_vec, self.h, self.solution_vec

    def generate_parameters(self):
        
        theta = np.random.multivariate_normal(mean=np.zeros(self.num_points),
                                              cov=self.covariance_matrix_theta)
        c = np.random.multivariate_normal(mean=np.zeros(self.num_points),
                                              cov=self.covariance_matrix_c)
                  
        u0 = np.random.multivariate_normal(mean=np.zeros(self.num_points),
                                              cov=self.covariance_matrix_u0)
                         
        for i in range(self.n_times-1):
            f = np.random.multivariate_normal(mean=np.zeros(self.num_points), cov=self.covariance_matrix_f)
            self.f_vec[i].x.array[:] = (f - np.min(f)) / (np.max(f) - np.min(f)) * 0.97 + 0.02 
                  
            g = np.random.multivariate_normal(mean=np.zeros(self.num_points), cov=self.covariance_matrix_g)
            self.g_vec[i].x.array[:] = (g - np.min(g)) / (np.max(g) - np.min(g)) * 0.97 + 0.02 

        self.theta.x.array[:] = (theta - np.min(theta)) / (np.max(theta) - np.min(theta)) * 0.97 + 0.02
        self.c.x.array[:] = (c - np.min(c)) / (np.max(c) - np.min(c)) * 0.97 + 0.02
        self.solution_vec[0].x.array[:] = (u0 - np.min(u0)) / (np.max(u0) - np.min(u0)) * 0.97 + 0.02
                  
        self.h = np.random.uniform(0.02, 0.99)
    
    
    def solve(self):
        # setup solver
        u_n = dolfinx.fem.Function(self.function_space)
        u_n.name = "u_n"
        u_n.x.array[:] = self.solution_vec[0].x.array[:] #np.zeros(self.num_points)
        
        u = ufl.TrialFunction(self.function_space)
        v = ufl.TestFunction(self.function_space)
                  
        ds = ufl.Measure("ds", domain=self.mesh)
        
        a = self.c * u * v * ufl.dx + self.dt * self.theta * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + self.h * self.dt * u * v * ds
        L = (self.c * u_n + self.dt * self.f) * v * ufl.dx + self.h * self.dt * self.g * v * ds

        bilinear_form = dolfinx.fem.form(a)
        linear_form = dolfinx.fem.form(L)

        A = dolfinx.fem.petsc.assemble_matrix(bilinear_form)
        A.assemble()
        b = dolfinx.fem.petsc.create_vector(linear_form)

        uh = dolfinx.fem.Function(self.function_space)

        #
        self.fem_solver = petsc4py.PETSc.KSP().create(self.mesh.comm)
        self.fem_solver.setOperators(A)
        self.fem_solver.setType(petsc4py.PETSc.KSP.Type.PREONLY)
        self.fem_solver.getPC().setType(petsc4py.PETSc.PC.Type.LU)
        #
        
        # solve
        for i in range(self.n_times-1):
                  
            self.f.x.array[:] = self.f_vec[i].x.array
            self.g.x.array[:] = self.g_vec[i].x.array
            
            # Update the right hand side reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            dolfinx.fem.petsc.assemble_vector(b, linear_form)
                
            # Solve linear problem
            self.fem_solver.solve(b, uh.x.petsc_vec)
            uh.x.scatter_forward()
        
            # Update solution at previous time step (u_n)
            u_n.x.array[:] = uh.x.array
            
            # Save solution at current time step
            self.solution_vec[i+1].x.array[:] = u_n.x.array  
        #endfor                
                                    

def generate_data(num_pdes, domain, function_space, param_pts, param_bry_pts, temp_pts, times, nb=0):
        
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
            

        # Find cells on which to evaluate param_pts and save them in cells_param_pts
        bb_tree = dolfinx.geometry.bb_tree(domain, domain.geometry.dim)
        potential_colliding_cells = dolfinx.geometry.compute_collisions_points(bb_tree, param_pts)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, potential_colliding_cells, param_pts)

        cells_param_pts = []
        for i in range(len(param_pts)):
            cells_param_pts.append(colliding_cells.links(i)[0])
        #endfor
                  
                  
        # Find cells on which to evaluate param_bry_pts and save them in cells_param_bry_pts
        bb_tree = dolfinx.geometry.bb_tree(domain, domain.geometry.dim)
        potential_colliding_cells = dolfinx.geometry.compute_collisions_points(bb_tree, param_bry_pts)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, potential_colliding_cells, param_bry_pts)

        cells_param_bry_pts = []
        for i in range(len(param_bry_pts)):
            cells_param_bry_pts.append(colliding_cells.links(i)[0])
        #endfor             
    
                  
            
        # Find cells on which to evaluate temp_pts and save them in cells_temp_pts
        bb_tree = dolfinx.geometry.bb_tree(domain, domain.geometry.dim)
        potential_colliding_cells = dolfinx.geometry.compute_collisions_points(bb_tree, temp_pts)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, potential_colliding_cells, temp_pts)

        cells_temp_pts = []
        for i in range(len(temp_pts)):
            cells_temp_pts.append(colliding_cells.links(i)[0])
        #endfor
            

        pde = pde_heat_eq__robin(domain, function_space, times)

        data = dict(
            f = np.zeros(shape = (num_pdes, len(times)-1, n_param_pts), dtype = np.float32),
            theta = np.zeros(shape = (num_pdes, n_param_pts), dtype = np.float32),
            c = np.zeros(shape = (num_pdes, n_param_pts), dtype = np.float32),
            g = np.zeros(shape = (num_pdes, len(times)-1, n_param_bry_pts), dtype = np.float32),
            h = np.zeros(shape = num_pdes, dtype = np.float32),
            solution = np.zeros(shape = (num_pdes, len(times), n_temp_pts), dtype = np.float32),
        )
        
        dir_name = lambda t: f"heat_eq_robin_2d_{num_pdes}/{t}"
        dir_record = lambda t: f"heat_eq_robin_2d_{num_pdes}/record_{t}.npy"

        os.system(f'rm -r {dir_name("")}')
        os.system(f'mkdir {dir_name("")}')
        # save mesh (not useful now)
        with dolfinx.io.XDMFFile(pde.mesh.comm, filename=dir_name("/mesh.xdmf"), file_mode="w") as file:
            file.write_mesh(pde.mesh)

        # save points and times #M in futuro si può aggiungere la possibilita' di valutare la sol in tempi diversi interpolando)
        with open(dir_name("eval_pts"), 'wb') as file:
            np.save(file, param_pts[:,:pde.tdim])
            np.save(file, param_bry_pts[:,:pde.tdim])
            np.save(file, temp_pts[:,:pde.tdim])
            np.save(file, times)
            np.save(file, nb)
            
        
        # save evaluations
        for i in range(num_pdes):
            if i%10==0: print("generating pde #", i)      
            pde()

            for j in range(len(times)-1):
                data["f"][i, j, :] = pde.f_vec[j].eval(x = param_pts, cells = cells_param_pts).flatten()
                data["g"][i, j, :] = pde.g_vec[j].eval(x = param_bry_pts, cells = cells_param_bry_pts).flatten()
            #endfor
                  
            for j in range(len(times)):
                data["solution"][i, j, :] = pde.solution_vec[j].eval(x = temp_pts, cells = cells_temp_pts).flatten()
            #endfor
                
            data["theta"][i, :] = pde.theta.eval(x = param_pts, cells = cells_param_pts).flatten()
            data["c"][i, :] = pde.c.eval(x = param_pts, cells = cells_param_pts).flatten()
            data["h"][i] = pde.h


            with open(dir_record(i), 'wb') as file:                  
                np.save(file, data["c"][i,:])
                np.save(file, data["theta"][i,:])
                np.save(file, data["f"][i, :, :])
                np.save(file, data["h"][i])
                np.save(file, data["g"][i, :, :])   
                np.save(file, data["solution"][i, :, :]) 
        #endfor
        

if __name__ == '__main__':
                                    
    # add or not points on the boundary
    boundary = False

    # number of pde instances to solve             
    num_pdes = 2000
                  

    # dolfinx mesh used to solve the pdes
    domain = dolfinx.mesh.create_rectangle(comm=MPI.COMM_WORLD, points=[[.0, .0], [1., 1.]], n=[16, 16], cell_type=dolfinx.mesh.CellType.triangle)
    function_space = dolfinx.fem.functionspace(domain, ("Lagrange", 1))
                  
                  
    #### Define points and times at which to evaluate the parameters and the solution ####

    # define evaluation points for parameters and temperature
    # param_points: nparray (n_param_points, 3) on which we sample the parameters (just a grid for now)
    # temp_points: nparray (n_temp_points, 3) on which we sample the temperature (a grid + boundary points if boundary==True)


    # times
    times = np.linspace(0,1,10)

    # grid
    n = 10
    v = np.linspace(0,1,n)
    grid_points = np.array([[v[i], v[j], 0] for j in range(n) for i in range(n)])
                  
                  
    # param boundary points
    m = 10
    w = np.linspace(0,1,m)

    bottom = np.column_stack((w, np.zeros(m)))
    top = np.column_stack((w, np.ones(m)))
    left = np.column_stack((np.zeros(m-2), w[1:-1]))
    right = np.column_stack((np.ones(m-2), w[1:-1]))

    param_bry_points = np.vstack((bottom, top, left, right))
    param_bry_points = np.column_stack((param_bry_points, np.zeros(param_bry_points.shape[0])))
    #print(param_bry_points.shape)              
                  

    # boundary points
    m = 57
    w = np.linspace(0,1,m)

    bottom = np.column_stack((w, np.zeros(m)))
    top = np.column_stack((w, np.ones(m)))
    left = np.column_stack((np.zeros(m-2), w[1:-1]))
    right = np.column_stack((np.ones(m-2), w[1:-1]))

    bry_points = np.vstack((bottom, top, left, right))
    bry_points = np.column_stack((bry_points, np.zeros(bry_points.shape[0])))
    #print(bry_points.shape)

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
    generate_data(num_pdes, domain, function_space, param_points, param_bry_points, temp_points, times, nb)
    print(f"\n\nData Generated - {datetime.now()} \n\n")        


        

