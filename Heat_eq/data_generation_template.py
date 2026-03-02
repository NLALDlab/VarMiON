
# SPDX-License-Identifier: GPL-3.0
# Copyright (c) 2025 NLALDlab
# Authors: NLALDlab, Marco Dell'Orto, Laura Rinaldi, Enrico Caregnato, Fabio Marcuzzi

"""
This script provides a template for generating a dataset for a PDE problem.

The dataset includes:
- Samples of PDE parameters
- Samples of initial and boundary conditions
- PDE solutions at sampled time-space points

The data is saved as '.npy' files, organized per PDE instance (in total 'num_pdes' files),
plus a shared 'eval_pts.npy' file that contains the coordinates and times used for sampling.

Sampling conventions:
- Functions defined over the full domain are sampled on a uniform NxN grid
- Boundary functions are sampled on M boundary points
- Time-dependent functions are sampled at 'k' time steps given in the array 'times'

The shared file `eval_pts.npy` stores the arrays containing the:
    - grid point coordinates                   shape: (N^2, d)
    - boundary point coordinates               shape: (M, d)
    - solution sampling points                 shape: (N^2 + nb, d)
    - time instants                            shape: (k,)

The i-th PDE instance is saved as `record_{i}.npy`, containing:
    - Parameter values, boundary/initial conditions, and the solution
"""

# As an example the script can be structured in the following way:

import numpy as np
import os

if __name__ == '__main__':
                   
    num_pdes = ...  # number of pde instances in the dataset                
    d = 2 # dimension of the domain
                      

    # --- Sampling points ---                                              
    param_pts = ...       # (N^2, d): points to sample parameters
    param_bry_pts = ...   # (M, d): points to sample boundary functions
    u_pts = ...           # (N^2 + nb, d): points to evaluate the solution
    times = ...           # (k,): time steps
    
    # === OUTPUT FOLDER SETUP ===
    dir_name = lambda t: f"dataset_{num_pdes}/{t}"
    os.system(f'rm -r {dir_name("")}')
    os.system(f'mkdir {dir_name("")}')
    
    # Save the shared evaluation points
    with open(dir_name("eval_pts"), 'wb') as file:
        np.save(file, param_pts)
        np.save(file, param_bry_pts)
        np.save(file, u_pts)
        np.save(file, times)
    
    # === DATA STORAGE STRUCTURE ===            
    data = dict(
        parameter_1 = np.zeros(shape = (num_pdes, len(times), len(param_pts)), dtype = np.float32),
        parameter_2 = np.zeros(shape = (num_pdes, len(param_bry_pts)), dtype = np.float32),
        # Add more as needed
        # ...
        solution = np.zeros(shape = (num_pdes, len(times), len(u_pts)), dtype = np.float32),
        ) 
                
    # === MAIN DATA GENERATION LOOP ===
    dir_record = lambda t: dir_name + f"record_{t}.npy"         
    for i in range(num_pdes):
        print("PDE instance #", i)      

        # --- Sample parameters and and initial/boundary conditions ---
        data["parameter_1"][i,:,:] = ... 
        data["parameter_2"][i,:] = ... 
        # Add other components here

        # --- Solve PDE for instance i ---
        data["solution"][i, :, :] = ...

        # --- Save the instance data --- 
        with open(dir_record(i), 'wb') as file:                  
            np.save(file, data["parameter_1"][i,:])
            np.save(file, data["parameter_2"][i,:])
            # Add more saves if you have more parameters

            np.save(file, data["solution"][i, :, :]) 
    #endfor
    
    print("Dataset generation complete.")
