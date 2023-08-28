import numpy as np

### Note this function return value in the exponential function of the prior
def prior_function_laplace(input_vector,mean_vector,variance_vector):

    ### Constant is moved based on the formulation of acceptance rate
    return -np.sum(np.abs(input_vector-mean_vector)/variance_vector)

def prior_function_spikeslab(input_vector,slab_variance):
    spike_variance = 1e-2
    slab_spike_samples = np.squeeze(np.random.binomial(1,0.5,6))
    # print(slab_spike_samples)
    
    num_ones = np.count_nonzero(slab_spike_samples)
    num_zero = 6 - num_ones
    
    constant = 1/np.power(np.sqrt(2*np.pi*spike_variance),num_zero)/np.power(np.sqrt(2*np.pi*slab_variance),num_ones)

    prior_wo_exp = 0

    for prior_sample in range(slab_spike_samples.shape[0]):
        if slab_spike_samples[prior_sample] == 1:
            prior_wo_exp = prior_wo_exp+(-np.square(input_vector[0,prior_sample])/2/slab_variance)
        else:
            prior_wo_exp = prior_wo_exp+(-np.square(input_vector[0,prior_sample])/2/spike_variance)
    
    return constant,prior_wo_exp

### Note this function return value in the exponential function of the likelihood
def likelihood(input_vector,mean_vector,covariance_matrix,parameter):
    num_para = parameter.shape[1]

    determinant = np.linalg.det(covariance_matrix)
    # print('determinant:',determinant)
    # print((input_vector-mean_vector))
    
    ### The constant part is removed based on the formulation in acceptance rate 
    return -0.5*(input_vector-mean_vector).T@np.linalg.inv(covariance_matrix)@(input_vector-mean_vector)

def Metropolis_Hasting(timestep,initial_sample,assumption_variance,databinding,prior_type):

    ### Prior info
    prior_mean = 0
    prior_variance = 1

    ### Release databinding
    Gdata = databinding[0]
    d_hat = databinding[1]
    covariance = databinding[2]

    num_parameters = Gdata.shape[1]

    ### Initialise MCMC
    post_sample_current = initial_sample             ### Initial samples for each datapoints
    post_sample_list = []      ### List of samples
    post_sample_list.append(post_sample_current)

    ### MCMC sampling
    for t in range(timestep):
        # print('Timestep:',t, end='\r')
        ### Important! The workflow below this is now univaraite!!! output [sample_size->1,num_vague]

        theta_new = np.random.multivariate_normal(np.squeeze(post_sample_current),\
                                                    np.diag(np.ones(num_parameters)*assumption_variance),1)
        

        ### build prior
        if prior_type == 'laplace':
            prior_function_upper = prior_function_laplace(theta_new,np.ones(num_parameters)*prior_mean,np.ones(num_parameters)*prior_variance)
            prior_function_lower = prior_function_laplace(post_sample_current,np.ones(num_parameters)*prior_mean,np.ones(num_parameters)*prior_variance)
        elif prior_type == 'spike-slab':
            cons_upper,prior_function_upper = prior_function_spikeslab(theta_new,prior_variance)
            cons_lower,prior_function_lower = prior_function_spikeslab(post_sample_current,prior_variance)

        Gtheta_upper = Gdata@theta_new.T
        Gtheta_lower = Gdata@post_sample_current.T

        ### Component to compute multivariate Gaussian function for likelihood 
        likelihood_upper = likelihood(Gtheta_upper,d_hat,covariance,theta_new)
        likelihood_lower = likelihood(Gtheta_lower,d_hat,covariance,post_sample_current)

        # print(prior_function_upper-prior_function_lower)
        # print(likelihood_upper-likelihood_lower)
        if prior_type == 'laplace':
            accept_ratio = np.squeeze(np.exp(prior_function_upper-prior_function_lower+likelihood_upper-likelihood_lower))
        elif prior_type == 'spike-slab':
            # print(cons_upper/cons_lower,np.exp(prior_function_upper-prior_function_lower),np.exp(likelihood_upper-likelihood_lower))
            accept_ratio = np.squeeze((cons_upper/cons_lower)*np.exp(prior_function_upper-prior_function_lower)*np.exp(likelihood_upper-likelihood_lower))
        
        
        check_sample = np.squeeze(np.random.uniform(0,1,1))

        if check_sample <= accept_ratio:
            post_sample_current = theta_new
            post_sample_list.append(post_sample_current)
            # print('Timestep: ',t,'; Accept ratio: ',accept_ratio,'; Xnew: ',theta_new,'; Accept')
            # print('Spike and slab:',spike_slab_upper)
        else:
            # print('Accept ratio: ',accept_ratio,'; Xnew: ',theta_new,'; Reject')
            pass
            
    
    
    # ### Truncate 1/4 of burning-in period sample
    truncate_num = int(len(post_sample_list)/4)
    print('Number of posterior samples: ',len(post_sample_list[truncate_num:]))
    print('The acceptance rate is: %', len(post_sample_list)/timestep*100)
    return np.squeeze(np.asarray(post_sample_list[truncate_num:]))