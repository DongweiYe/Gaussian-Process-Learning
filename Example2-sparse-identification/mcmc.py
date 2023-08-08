import numpy as np

def prior_function(input_vector,mean_vector,variance_vector):

    return 1/(np.prod(2*variance_vector))*np.exp(-np.sum(np.abs(input_vector-mean_vector)/variance_vector))

def likelihood(input_vector,mean_vector,covariance_matrix,parameter,L1regu=False):
    num_para = parameter.shape[1]

    determinant = np.linalg.det(covariance_matrix)
    # print('determinant:',determinant)
    # print((input_vector-mean_vector))
    if L1regu == False:
        return 1/np.sqrt((2*np.pi)**num_para*determinant)*\
                                np.exp(-0.5*(input_vector-mean_vector).T@np.linalg.inv(covariance_matrix)@(input_vector-mean_vector))
    else:
        return 1/np.sqrt((2*np.pi)**num_para*determinant)*\
                                np.exp(-0.5*(input_vector-mean_vector).T@np.linalg.inv(covariance_matrix)@(input_vector-mean_vector)) + 2*L1regu*np.sum(np.abs(parameter))


def Metropolis_Hasting(timestep,initial_sample,assumption_variance,databinding,L1regu):

    ### Prior info
    prior_mean = 0
    prior_variance = 4

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
        
        # theta_new= np.array([[0,1.5,0,0,0,-1]])
        ### build prior
        prior_function_upper = prior_function(theta_new,np.ones(num_parameters)*prior_mean,np.ones(num_parameters)*prior_variance)
        prior_function_lower = prior_function(post_sample_current,np.ones(num_parameters)*prior_mean,np.ones(num_parameters)*prior_variance)

        Gtheta_upper = Gdata@theta_new.T
        Gtheta_lower = Gdata@post_sample_current.T

        ### Component to compute multivariate Gaussian function for likelihood 
        likelihood_upper = likelihood(Gtheta_upper,d_hat,covariance,theta_new,L1regu=L1regu)
        likelihood_lower = likelihood(Gtheta_lower,d_hat,covariance,post_sample_current,L1regu=L1regu)

        # print(prior_function_upper,prior_function_lower)
        # print(likelihood_upper,likelihood_lower)

        upper_alpha = prior_function_upper*likelihood_upper
        lower_alpha = prior_function_lower*likelihood_lower

        accept_ratio = np.squeeze(upper_alpha/lower_alpha) 
        check_sample = np.squeeze(np.random.uniform(0,1,1))

        if check_sample <= accept_ratio:
            post_sample_current = theta_new
            post_sample_list.append(post_sample_current)
            # print('Accept ratio: ',accept_ratio,'; Xnew: ',theta_new,'; Accept')
        else:
            pass
            # print('Accept ratio: ',accept_ratio,'; Xnew: ',theta_new,'; Reject')
    
    
    # ### Truncate 1/4 of burning-in period sample
    truncate_num = int(len(post_sample_list)/4)
    print('Number of posterior samples: ',len(post_sample_list[truncate_num:]))
    print('The acceptance rate is: %', len(post_sample_list[truncate_num:])/timestep*100)
    return np.squeeze(np.asarray(post_sample_list[truncate_num:]))