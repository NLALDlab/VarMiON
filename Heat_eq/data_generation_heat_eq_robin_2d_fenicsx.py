# SPDX-License-Identifier: GPL-3.0
# Copyright (c) 2025 NLALDlab
# Authors: NLALDlab, Marco Dell'Orto, Laura Rinaldi, Enrico Caregnato, Fabio Marcuzzi


import os
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from tqdm import tqdm
import shutil
import argparse


# works with dolfinx 0.9.0
import dolfinx, dolfinx.fem.petsc, petsc4py, ufl
import mpi4py.MPI as MPI



# contains the classes:
# PDE --> gives the structure of a class that solves a PDE
# pde_heat_eq__robin(PDE) --> solves the heat equation with Robin boundary conditions


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


 
                  
class pde_heat_eq__robin(PDE):
    def __init__(self, mesh, function_space, times):
        super().__init__(mesh=mesh, function_space=function_space)
 
        self.t = times[0]
        self.T = times[-1]
        self.n_times = len(times)
        self.dt = times[1] - times[0]

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
        self.g.name = 'g'  
                  
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
        u_n.x.array[:] = self.solution_vec[0].x.array[:]
        
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

        
        self.fem_solver = petsc4py.PETSc.KSP().create(self.mesh.comm)
        self.fem_solver.setOperators(A)
        self.fem_solver.setType(petsc4py.PETSc.KSP.Type.PREONLY)
        self.fem_solver.getPC().setType(petsc4py.PETSc.PC.Type.LU)
        
        
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
        
        
        
def evaluation_cells(domain, pts):

    # Find cell indices on which to evaluate the given points

    # Inputs:
    # domain: dolfinx mesh
    # pts: nparray with shape (number of points, 3)
    
    # Output:
    # cells: list of indices such that the point pts[i,:] belongs to the cell cells[i] of domain
    
    bb_tree = dolfinx.geometry.bb_tree(domain, domain.geometry.dim)
    potential_colliding_cells = dolfinx.geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, potential_colliding_cells, pts)

    cells = []
    for i in range(len(pts)):
        cells.append(colliding_cells.links(i)[0])
    #endfor
    return cells
        
                                    

def generate_data(num_pdes, domain, function_space, param_pts, param_bry_pts, temp_pts, times):
        
        # num_pdes: number of pde instances to be generated
        # param_pts: nparray with shape (n_param_pts, 3) containing points on which we sample parameters
        # param_bry_pts: nparray with shape (n_param_bry_pts, 3) containing boundary points on which we sample parameters
        # temp_pts: nparray with shape (n_temp_pts, 3), spatial grid on which we sample the temperature

        n_param_pts = len(param_pts)
        n_param_bry_pts = len(param_bry_pts)
        n_temp_pts = len(temp_pts)

        # Reorder temp_pts
        indices = np.lexsort(temp_pts.T)
        temp_pts = temp_pts[indices]
        
        
        
        cells_param = evaluation_cells(domain, param_pts)
        cells_param_bry = evaluation_cells(domain, param_bry_pts)
        cells_temp = evaluation_cells(domain, temp_pts)

                       
        # create PDE object
        pde = pde_heat_eq__robin(domain, function_space, times)

        
        data = dict(
            f = np.zeros(shape = (num_pdes, len(times)-1, n_param_pts), dtype = np.float32),
            theta = np.zeros(shape = (num_pdes, n_param_pts), dtype = np.float32),
            c = np.zeros(shape = (num_pdes, n_param_pts), dtype = np.float32),
            g = np.zeros(shape = (num_pdes, len(times)-1, n_param_bry_pts), dtype = np.float32),
            h = np.zeros(shape = num_pdes, dtype = np.float32),
            solution = np.zeros(shape = (num_pdes, len(times), n_temp_pts), dtype = np.float32),
        )

        
        # create directory in which to save the data
        # if there already exists a directory with the same name, delete it after user confirmation
        dir_name = f"heat_eq_robin_2d_{num_pdes}"
        dir_record = lambda t: f"{dir_name}/record_{t}.npy"
        
        
        while os.path.isdir(dir_name):
            confirm = input(f"\nA folder with the name '{dir_name}' already exists. \nProceeding will permanently delete its contents. Continue? (y/n): ").strip().lower()
            if confirm == "y":
                shutil.rmtree(dir_name)
                print("\nFolder deleted.\n")
            else:
                print("\nThe folder has not been deleted.\n")
                dir_name = input(f"Enter a name for the new dataset folder: ").strip()
            #endif
        #endwhile
            
        
       
        #if os.path.isdir(dir_name):
        #    print(f"A folder with name {dir_name} already exists.")
        #    confirm = input(f"Delete existing dataset in '{dir_name}'? (y/n): ").strip().lower()
        #    if confirm == "y":
        #        shutil.rmtree(dir_name)
        #        print("\nPrevious dataset deleted.\nGenerating new dataset.\n")
        #    #endif
        #    else:
        #        print("\nExisting dataset has not been deleted.\n")
        #        dir_name = input(f"Enter a name for the new dataset folder (existing folders may be overwritten): ").strip()
        #        shutil.rmtree(dir_name)                
        ##endif

        os.system(f'mkdir {dir_name}')

        # save points and times
        with open(dir_name + "/eval_pts", 'wb') as file:
            np.save(file, param_pts[:,:pde.tdim])
            np.save(file, param_bry_pts[:,:pde.tdim])
            np.save(file, temp_pts[:,:pde.tdim])
            np.save(file, times)
            
        
        print("\n Generating dataset...\n")
        for i in tqdm(range(num_pdes)):
            
            pde()

            for j in range(len(times)-1):
                data["f"][i, j, :] = pde.f_vec[j].eval(x = param_pts, cells = cells_param).flatten()
                data["g"][i, j, :] = pde.g_vec[j].eval(x = param_bry_pts, cells = cells_param_bry).flatten()
            #endfor
                  
            for j in range(len(times)):
                data["solution"][i, j, :] = pde.solution_vec[j].eval(x = temp_pts, cells = cells_temp).flatten()
            #endfor
                
            data["theta"][i, :] = pde.theta.eval(x = param_pts, cells = cells_param).flatten()
            data["c"][i, :] = pde.c.eval(x = param_pts, cells = cells_param).flatten()
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_pdes', default=2000, type=int)
    args = parser.parse_args() 

    # number of pde instances to solve             
    num_pdes = args.num_pdes
                  

    # dolfinx mesh used to solve the pdes
    domain = dolfinx.mesh.create_rectangle(comm=MPI.COMM_WORLD, points=[[.0, .0], [1., 1.]], n=[16, 16], cell_type=dolfinx.mesh.CellType.triangle)
    function_space = dolfinx.fem.functionspace(domain, ("Lagrange", 1))
                  
                  
    #### Define points and times at which to evaluate the parameters and the solution ####


    # num_pdes: number of pde instances to be generated
    # param_pts: nparray with shape (n_param_pts, 3) containing points on which we sample parameters
    # param_bry_pts: nparray with shape (n_param_bry_pts, 3) containing boundary points on which we sample parameters
    # temp_pts: nparray with shape (n_temp_pts, 3), spatial grid on which we sample the temperature


    # times
    times = np.linspace(0,.1,10)

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


    temp_points = grid_points


    print(f"\n\nData Generation - {datetime.now()} \n\n")
    
    generate_data(num_pdes, domain, function_space, param_points, param_bry_points, temp_points, times)
    
    print(f"\n\nData Generated - {datetime.now()} \n\n")        


        

