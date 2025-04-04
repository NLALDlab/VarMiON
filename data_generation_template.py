# this file should generate a folder containing:

# an npy file 'eval_pts' in which there are the numpy arrays:

    # an array with shape (100, 2) containing the coordinates of the points on which we sample the parameters C, theta and f 
    # an array with shape (36, 2) containing the coordinates of the points on the boundary on which we sample g
    # an array with shape (n_times,) containing the time instants at which we compute the solution, including the first time instant for which the solution is given
    # an array with shape () containing an integer that specifies the number of boundary points on which we sample the solution


# for each pde instance a file 'record_{i}.npy' with i=0,...,num_pdes-1 with:
    # an array with shape (100,) containing a sample of the thermal capacity C
    # an array with shape (100,) containing a sample of the thermal conductivity theta
    # an array with shape (n_times-1, 100) containing a sample of the heat source f
    # an array with shape () containing the value of the heat tranfer constant h
    # an array with shape (n_times-1, 36) containing a sample of the environment temperature g
    # an array with shape (n_times, 100) containing a sample of the solution
