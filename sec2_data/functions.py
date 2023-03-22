import numpy as np
import json

from scipy import interpolate
from scipy import misc

##--- Concerning the addition of derivatives to a dataset ---##  

def add_derivatives(input_data, base_features, t_steps=None, h_fields=None, new_features=None, loc=None, order=2):
    '''
        Adds derivatives to the input_data
        
        ndarray input_data: Matrix of shape (no_features, no_samples) or dict
        dict loc: To locate variables (not needed for the dict)
        arr base_features: Old basic feature names
        arr new_features: New feature names (not needed for the dict)
    '''
    # Try treating the input as a dict
    try:        
        # Process all time steps/horizontal fields if not specified differently
        if t_steps != None:
            t_steps_loop = t_steps
        else:
            t_steps_loop = input_data['zg'].shape[0]

        if h_fields != None:
            h_fields_loop = h_fields
        else:
            h_fields_loop = input_data['zg'].shape[2]

        for var in base_features[:-1]:
            input_data[var+'_z'] = 1000*np.ones((t_steps_loop, 27, h_fields_loop))
            if order == 2:
                input_data[var+'_zz'] = 1000*np.ones((t_steps_loop, 27, h_fields_loop))

        for T in range(t_steps_loop):
            for H in range(h_fields_loop):
                # Interpolate y as a function of x and differentiate in x0
                x = input_data['zg'][T, :, H]
                # Shift upper- and lower-most level by 1m so that the derivatives are well-defined
                x0 = x.copy(); x0[0] = x0[0] - 1; x0[-1] = x0[-1] + 1

                for var in base_features[:-1]:
                    y = input_data[var][T, :, H]
                    f_z, f_zz = differentiate_multiple(x, y, x0, order)
                    input_data[var+'_z'][T, :, H] = f_z
                    if order == 2:
                        input_data[var+'_zz'][T, :, H] = f_zz

        ## Double-check
        # assert np.all(np.abs(input_data[base_features[0]+'_z'] - 1000) > 1e-6)
        return input_data
    
    # If the input is a numpy array
    except:
        print('An exception occurred: The argument is not treated as a dictionary but as a np array instead.')
        n = input_data.shape[1]
        input_aug = 1000*np.ones((n, 27, len(new_features)))

        for sample in range(n):
            i = 0
            # Interpolate y as a function of x and differentiate in x0
            x = input_data[loc['zg_21']:loc['zg_47']+1, sample]
            # Shift upper- and lower-most level by 1m so that the derivatives are well-defined
            x0 = x.copy(); x0[0] = x0[0] - 1; x0[-1] = x0[-1] + 1
            for var in base_features[:-1]:
                y = input_data[loc[var+'_21']:loc[var+'_47']+1, sample]
                f_z, f_zz = differentiate_multiple(x, y, x0)
                input_aug[sample, :, i] = y
                input_aug[sample, :, i+1] = f_z
                input_aug[sample, :, i+2] = f_zz
                i = i + 3
            input_aug[sample, :, i] = x
            input_aug[sample, :, i+1] = input_data[loc['fr_land'], sample]

        ## Double-check
        assert np.all(np.abs(input_aug - 1000) > 1e-6)

        return input_aug

def differentiate(x, y, x0):
    '''
        A function f: x -> y is fitted and its derivatives f'(x0) and f''(x0) are evaluated
        
        arr x: Domain
        arr y: Codomain
        float x0: Where to evaluate the derivative
    '''
    assert len(x) == len(y)

    ## The output depends slightly on x0 and the interpolator, is independent of the order of approximation
    # Choices: linear, quadratic, cubic
    f = interpolate.interp1d(x, y, kind='cubic')

    # Take the derivative in x0
    f_z = misc.derivative(f, x0, dx=1e-2, n=1, args=(), order=3)
    f_zz = misc.derivative(f, x0, dx=1e-2, n=2, args=(), order=5)

    return (f_z, f_zz)

def differentiate_multiple(x, y, x0, order=2):
    '''
        A function f: x -> y is fitted and its derivatives f'(w) and f''(w) are evaluated for all w in x0
        
        arr x: Domain
        arr y: Codomain
        arr x0: Where to evaluate the derivative
    '''
    assert len(x) == len(y)

    ## The output depends slightly on x0 and the interpolator, is independent of the order of approximation
    # Choices: linear, quadratic, cubic
    f = interpolate.interp1d(x, y, kind='cubic')

    # Take the derivatives in w
    f_z = [misc.derivative(f, w, dx=1e-2, n=1, args=(), order=3) for w in x0]
    if order == 2:
        f_zz = [misc.derivative(f, w, dx=1e-2, n=2, args=(), order=5) for w in x0]
    else:
        f_zz = -1000

    return (f_z, f_zz)


##--- Concerning the evaluation of the Sundqvist Scheme ---##

def evaluate_sundqvist(input_data, output_data, loc, tuned, best_land=None, best_sea=None, compute_r2=True):
    '''
        Evaluates the output of Sundqvist's scheme against the provided output data.
        Here we use the best set of hyperparameters/tuning parameters that we found earlier.
        
        ndarray input_data: Has shape (samples x features)
        ndarray output_data: Has shape (samples) and values in [0, 100] 
        dict loc: To locate columns of input_data corresponding to specific variables
        array tuned: Whether the manually or automatically tuned hyperparameters or the original ones should be used. 
                     If tuned == 'custom', then we have to supply the hyperparameters best_land and best_sea.
    '''
    if tuned=='manually':
        rsat_best_land = 1.1
        r0_top_best_land = 0.2
        r0_surf_best_land = 0.85
        n_best_land = 1.62

        rsat_best_sea = 1
        r0_top_best_sea = 0.34
        r0_surf_best_sea = 0.95
        n_best_sea = 1.35
    elif tuned=='automatically':
        rsat_best_land = 1.478
        r0_top_best_land = 0.005
        r0_surf_best_land = 0.494
        n_best_land = 1.114

        rsat_best_sea = 1.425
        r0_top_best_sea = 0.039
        r0_surf_best_sea = 0.765
        n_best_sea = 1.283
    elif tuned=='original':
        rsat_best_land = 1
        r0_top_best_land = 0.8
        r0_surf_best_land = 0.968
        n_best_land = 2
        
        rsat_best_sea = rsat_best_land
        r0_top_best_sea = r0_top_best_land
        r0_surf_best_sea = r0_surf_best_land
        n_best_sea = n_best_land
    elif tuned=='custom':
        rsat_best_land, r0_top_best_land, r0_surf_best_land, n_best_land = best_land
        rsat_best_sea, r0_top_best_sea, r0_surf_best_sea, n_best_sea = best_sea
    
    no_samples = input_data.shape[0]
    
    # Computes the mse on the training set
    mse = 0
    mse_exclude_upper = 0
    output_exclude_upper = []
    
    for i in range(no_samples):
        ps = input_data[i, loc['ps']]
        try:
            p = input_data[i, loc['pres']]
        except:
            p = input_data[i, loc['pa']]
        r = input_data[i, loc['rh']]
        
        # Differentiate between land and sea
        fr_land = input_data[i, loc['fr_land']]
        if fr_land > 0.5:
            rsat = rsat_best_land
            r0_top = r0_top_best_land
            r0_surf = r0_surf_best_land
            n = n_best_land
        else:
            rsat = rsat_best_sea
            r0_top = r0_top_best_sea
            r0_surf = r0_surf_best_sea
            n = n_best_sea
            
        r0 = r0_top + (r0_surf - r0_top)*np.exp(1-(ps/p)**n)
        if r > r0:
            # r can theoretically exceed rsat
            c = 1 - np.sqrt((np.minimum(r, rsat) - rsat)/(r0 - rsat)) # in [0,1]
        else:
            c = 0
        mse = mse + (100*c - output_data[i])**2
        
        # Extract mse exluding the upper-most two layers
        if compute_r2 and not i%27 in [0, 1]:
            mse_exclude_upper = mse_exclude_upper + (100*c - output_data[i])**2
            output_exclude_upper.append(output_data[i])
        
    # Final MSE score
    mse = mse/no_samples
    # R2 score. Do not compute it for the upper-most two layers!
    if compute_r2:
        mse_exclude_upper = mse_exclude_upper/len(output_exclude_upper)
        var = np.var(output_exclude_upper)
        r2 = 1-mse_exclude_upper/var
        return mse, r2
    
    return mse

def evaluate_sample_sundqvist(input_sample, output_sample, loc, tuned):
    '''
        Evaluates the output of Sundqvist's scheme against the provided output data of one grid cell.
        Here we use the best set of hyperparameters/tuning parameters that we found earlier.
        
        ndarray input_sample: Has shape (features)
        float output_sample: Has values in [0, 100] 
        dict loc: To locate columns of input_sample corresponding to specific variables
        array tuned: Whether the manually or automatically tuned hyperparameters or the original ones should be used 
    '''
    if tuned=='manually':
        rsat_best_land = 1.1
        r0_top_best_land = 0.2
        r0_surf_best_land = 0.85
        n_best_land = 1.62

        rsat_best_sea = 1
        r0_top_best_sea = 0.34
        r0_surf_best_sea = 0.95
        n_best_sea = 1.35
    elif tuned=='automatically':
        rsat_best_land = 1.478
        r0_top_best_land = 0.005
        r0_surf_best_land = 0.494
        n_best_land = 1.114

        rsat_best_sea = 1.425
        r0_top_best_sea = 0.039
        r0_surf_best_sea = 0.765
        n_best_sea = 1.283
    elif tuned=='original':
        rsat_best_land = 1
        r0_top_best_land = 0.8
        r0_surf_best_land = 0.968
        n_best_land = 2
        
        rsat_best_sea = rsat_best_land
        r0_top_best_sea = r0_top_best_land
        r0_surf_best_sea = r0_surf_best_land
        n_best_sea = n_best_land
    
    no_samples = input_sample.shape[0]
    
    # Computes the mse on the training set
    ps = input_sample[loc['ps']]
    p = input_sample[loc['pres']]
    r = input_sample[loc['rh']]
    
    # Differentiate between land and sea
    fr_land = input_sample[loc['fr_land']]
    if fr_land > 0.5:
        rsat = rsat_best_land
        r0_top = r0_top_best_land
        r0_surf = r0_surf_best_land
        n = n_best_land
    else:
        rsat = rsat_best_sea
        r0_top = r0_top_best_sea
        r0_surf = r0_surf_best_sea
        n = n_best_sea
    	
    r0 = r0_top + (r0_surf - r0_top)*np.exp(1-(ps/p)**n)
    if r > r0:
        # r can theoretically exceed rsat
        c = 1 - np.sqrt((np.minimum(r, rsat) - rsat)/(r0 - rsat)) # in [0,1]
    else:
        c = 0
        
    mae = np.abs(100*c - output_sample)
    mse = (100*c - output_sample)**2
    
    return mse, mae, 100*c


##--- Concerning the search of symmetries with the help of our NNs from the JAMES paper ---##

def draw_uniform(a, b):
    '''
        We uniformly draw a sample from the interval [a, b]
    '''
    assert a < b
    return np.random.random(1)*(b-a)+a

def draw_uniform_intersect_intervals(a1, b1, a2, b2):
    '''
        We uniformly draw a sample from the intersection of these two intervals [a1, b1] and [a2, b2]
    '''
    assert a1 < b1 and a2 < b2
    if a1 <= a2 <= b1 <= b2:
        return draw_uniform(a2, b1)
    if a1 <= a2 <= b2 <= b1:
        return draw_uniform(a2, b2)
    if a2 <= a1 <= b1 <= b2:
        return draw_uniform(a1, b1)
    if a2 <= a1 <= b2 <= b1:
        return draw_uniform(a1, b2)
    
def draw_uniform_union_intervals(a1, b1, a2, b2):
    '''
        We uniformly draw a sample from the union of these two intervals [a1, b1] and [a2, b2]
    '''
    assert a1 < b1 or a2 < b2
    if b1 < a2 or b2 < a1:
        # Depending on the length of the two intervals we draw either from the one or the other
        c = draw_uniform(np.minimum(-(b2-a2), 0), np.maximum(b1-a1, 0))
        if np.sign(c) == 1:
            return draw_uniform(a1, b1)
        else:
            return draw_uniform(a2, b2)
    else:
        return draw_uniform(np.minimum(a1, a2), np.maximum(b1, b2))    
    
def draw_scaling_factors(mu_x, mu_y, sigma_x, sigma_y):
    '''
        We draw three scaling factors, that can be either added to mu_x and mu_y (or subtracted, multiplied or divided by).
        The result would still be in the range of values the NN is comfortable (within a std of the mean value)
        This function is used and described in ./finding_symmetries.
    '''    
    # One variable suffices for both translational symmetries
    std_min = np.minimum(sigma_x, sigma_y)
    a_t = draw_uniform(-std_min, std_min)
    
    # For the first scaling symmetry
    frac_min_1 = np.minimum(sigma_x/np.abs(mu_x), sigma_y/np.abs(mu_y))
    a_s_1 = draw_uniform(1-frac_min_1, 1+frac_min_1)
    
    # For the second scaling symmetry
    frac_min_2 = sigma_y/np.abs(mu_y)
    b_x = 1 - np.sign(mu_x)*sigma_x/mu_x
    c_x = 1 + np.sign(mu_x)*sigma_x/mu_x

    if np.sign(b_x) == np.sign(c_x):
        a_s_2 = draw_uniform_intersect_intervals(1/c_x, 1/b_x, 1-frac_min_2, 1+frac_min_2)
    else:
        # Draw from the intersection of [-infty, 1/b_x] and [1-frac_min_2, 1+frac_min_2] 
        # Or from the intersection of [1/c_x, infty] and [1-frac_min_2, 1+frac_min_2]
        # Otherwise from the union of [1-frac_min_2, 1/b_x] and [1/c_x, 1+frac_min_2]
        if 1+frac_min_2 < 1/b_x:
            a_s_2 = draw_uniform_intersect_intervals(-10**8, 1/b_x, 1-frac_min_2, 1+frac_min_2)
        if 1-frac_min_2 > 1/c_x:
            a_s_2 = draw_uniform_intersect_intervals(1/c_x, 10**8, 1-frac_min_2, 1+frac_min_2)
        else:
            a_s_2 = draw_uniform_union_intervals(1-frac_min_2, 1/b_x, 1/c_x, 1+frac_min_2)
            
    # Add assertions to check whether drawn samples are really in the right range!
    assert mu_x-sigma_x <= mu_x+a_t <= mu_x+sigma_x and mu_y-sigma_y <= mu_y+a_t <= mu_y+sigma_y
    assert mu_x-sigma_x <= mu_x-a_t <= mu_x+sigma_x and mu_y-sigma_y <= mu_y-a_t <= mu_y+sigma_y
    assert mu_x-sigma_x <= mu_x*a_s_1 <= mu_x+sigma_x and mu_y-sigma_y <= mu_y*a_s_1 <= mu_y+sigma_y
    assert mu_x-sigma_x <= mu_x/a_s_2 <= mu_x+sigma_x and mu_y-sigma_y <= mu_y*a_s_2 <= mu_y+sigma_y
    
    return a_t, a_s_1, a_s_2


##--- Appending a dictionary to a JSON file ---##

def append_dict_to_json(d, outfile):
    '''
        d: dictionary
        outfile: path to the json-file
    '''
    # Write dictionary to json file
    try:
        with open(outfile, 'r') as file:
            read_dict = json.load(file)
            read_dict.update(d)
        with open(outfile, 'w') as file:
            json.dump(read_dict, file)
    except(json.JSONDecodeError, FileNotFoundError):
        with open(outfile, 'w') as file:
            json.dump(d, file)
        print('New file created or first entry added')
        
##--- Helper functions ---##

def describe(arr):
    '''
        Descriptive statistics of a numpy array
    '''
    # measures of central tendency
    mean = np.mean(arr)
    median = np.median(arr)

    # measures of dispersion
    min = np.amin(arr)
    max = np.amax(arr)
    range = np.ptp(arr)
    variance = np.var(arr)
    sd = np.std(arr)

    print("Descriptive analysis")
    print("Array =", arr)
    print("Measures of Central Tendency")
    print("Mean =", mean)
    print("Median =", median)
    print("Measures of Dispersion")
    print("Minimum =", min)
    print("Maximum =", max)
    print("Range =", range)
    print("Variance =", variance)
    print("Standard Deviation =", sd)