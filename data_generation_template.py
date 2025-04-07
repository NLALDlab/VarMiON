# This file should generate a folder containing:

# an npy file 'eval_pts' in which there are:

    # one nparray with shape (100, 2) containing the coordinates of the points where we sample the parameters C, theta and f 
    # one nparray with shape (36, 2) containing the coordinates of the points on the boundary where we sample g
    # one nparray with shape (100+nb,) containing the coordinates of the points where we sample the temperature
    # one nparray with shape (n_times,) containing the time instants where we compute the solution, including the first time instant for which the solution is given
    # one nparray with shape () containing an integer that specifies the number of boundary points where we sample the solution


# and for each pde instance a file 'record_{i}.npy' with i=0,...,num_pdes-1 with:
    # an array with shape (100,) containing a sample of the thermal capacity C
    # an array with shape (100,) containing a sample of the thermal conductivity theta
    # an array with shape (n_times-1, 100) containing a sample of the heat source f
    # an array with shape () containing the value of the heat tranfer constant h
    # an array with shape (n_times-1, 36) containing a sample of the environment temperature g
    # an array with shape (n_times, 100) containing a sample of the solution


# As an example the script can be structured in the following way:

if __name__ = '__main__':
                   
    num_pdes = ...  # number of pde instances in the dataset                
    pde.tdim = 2 # dimension of the domain                  
                                                    
    param_pts = ...  # coordinates of the points where we sample the parameters C, theta and f
    param_bry_pts = ... # containing the coordinates of the points on the boundary where we sample g
    temp_pts = ... # coordinates of the points where we sample the temperature
    times = ... # time instants where we compute the solution, including the first time instant for which the solution is given
    nb = 0 # number of extra boundary points where we sample the temperature            #M da togliere?
    
    
    # create a folder in which to save the results
    dir_name = lambda t: f"heat_eq_robin_2d_{num_pdes}/{t}"
    os.system(f'rm -r {dir_name("")}')
    os.system(f'mkdir {dir_name("")}')
    
    with open(dir_name("eval_pts"), 'wb') as file:
        np.save(file, param_pts)
        np.save(file, param_bry_pts)
        np.save(file, temp_pts)
        np.save(file, times)
        np.save(file, nb)
    
                  
    # we define a dictionary where we temporarily store the data            
    data = dict(
        f = np.zeros(shape = (num_pdes, len(times)-1, n_param_pts), dtype = np.float32),
        theta = np.zeros(shape = (num_pdes, n_param_pts), dtype = np.float32),
        c = np.zeros(shape = (num_pdes, n_param_pts), dtype = np.float32),
        g = np.zeros(shape = (num_pdes, len(times)-1, n_param_bry_pts), dtype = np.float32),
        h = np.zeros(shape = num_pdes, dtype = np.float32),
        solution = np.zeros(shape = (num_pdes, len(times), n_temp_pts), dtype = np.float32),
        ) 
                
    
    dir_record = lambda t: f"heat_eq_robin_2d_{num_pdes}/record_{t}.npy"
              
    for i in range(num_pdes):
        print("pde instance #", i)      
        
        ### generate the parameters and boundary/initial conditions relative to the i-th instance of the pde
        for j in range(len(times)-1):
            data["f"][i, j, :] = ...
            data["g"][i, j, :] = ...
        #endfor
            
        data["theta"][i, :] = ...
        data["c"][i, :] = ...
        data["h"][i] = ...
              
        ### solve instance i of the pde obtaining the solution:     
        for j in range(len(times)):
            data["solution"][i, j, :] = ...
        #endfor

        # store the data in a npy file 
        with open(dir_record(i), 'wb') as file:                  
            np.save(file, data["c"][i,:])
            np.save(file, data["theta"][i,:])
            np.save(file, data["f"][i, :, :])
            np.save(file, data["h"][i])
            np.save(file, data["g"][i, :, :])   
            np.save(file, data["solution"][i, :, :]) 
    #endfor
