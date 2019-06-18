#!/usr/bin/env python

# Light curve tools
# ---
# Abhimat Gautam

import numpy as np

def lc_polyfit(mags, mag_errs, obs_times, sig_test=5.):
    detected_filter = np.where(mags > 0.0)
    
    # Filter out to non-detected epochs
    mags_det = mags[detected_filter]
    mag_errs_det = mag_errs[detected_filter]
    obs_times_det = obs_times[detected_filter]
    
    num_obs = len(mags_det)
    
    # Calculate mean mag
    obs_weights = 1./(mag_errs_det **2.)
    mag_mean = (np.sum(obs_weights * mags_det) /
                np.sum(obs_weights))
    
    # Calculate reduced chi squared for different order models
    from scipy.special import erf
    sig_test_var_prob = erf(sig_test / np.sqrt(2.))
    
    ## 0th order
    num_params = 1
    dof = float(num_obs - num_params)
    
    model = (obs_times_det * 0.) + mag_mean
    
    poly_coeffs = np.array([mag_mean])
    
    red_chiSq = (1./dof) * np.sum(((mags_det - model)**2.) /
                                  (mag_errs_det)**2.)
    
    mags_polyfit = mags_det
    
    highest_D_val_prob = sig_test_var_prob
    
    ## 1st and 2nd order
    for trial_deg in [1, 2]:
        cur_num_params = trial_deg + 1
        cur_poly_coeffs = np.polyfit(obs_times_det, mags_det,
                                 trial_deg, w=(1./mag_errs_det))
        
        cur_dof = float(num_obs - cur_num_params)
        cur_model = np.polyval(cur_poly_coeffs, obs_times_det)
        
        cur_red_chiSq = (1./cur_dof) * np.sum(((mags_det - cur_model)**2.) /
                                              (mag_errs_det)**2.)
        
        cur_D_val_prob = wilks_theorem_test(mags_det, mag_errs_det,
                                            model, num_params,
                                            cur_model, cur_num_params)
        
        if cur_D_val_prob >= highest_D_val_prob:
            num_params = cur_num_params
            dof = cur_dof
            model = cur_model
            poly_coeffs = cur_poly_coeffs
            red_chiSq = cur_red_chiSq
            
            mags_polyfit = (mags_det - model) + mag_mean
            
            highest_D_val_prob = cur_D_val_prob
        
    
    return (mags_polyfit, model, poly_coeffs, dof, red_chiSq)
    

def log_like(mags, mag_errs, model_mags):
    """Return log likelihood
    
    Keyword arguments:
    mags
    mag_errs
    model_mags
    """
    
    log_like = 0.
    
    log_like += -0.5 * np.sum(np.log(2. * np.pi * mag_errs**2.))
    
    log_like += -0.5 * np.sum((mags - model_mags)**2. / mag_errs**2.)
    
    return log_like
    

def wilks_theorem_test(mags, mag_errs,
                       mags_mod_null, num_params_mod_null,
                       mags_mod_alt, num_params_mod_alt):
    # Calculate degrees of freedom
    df_mod_null = len(mags) - num_params_mod_null
    df_mod_alt = len(mags) - num_params_mod_alt
    
    # Calculate log likelihoods for null and alt models
    log_like_mod_null = log_like(mags, mag_errs, mags_mod_null)
    log_like_mod_alt = log_like(mags, mag_errs, mags_mod_alt)
    
    # Calculate D val
    D_val = 2. * (log_like_mod_alt - log_like_mod_null)
    
    # Calculate probability with chi2
    from scipy.stats import chi2
    
    chi2_df = num_params_mod_alt - num_params_mod_null
    
    chi2_cdf = chi2.cdf(D_val, chi2_df)
    
    return chi2_cdf

def lc_polyfit_tester():
    from gc_photdata import align_dataset
    
    # Read in saved data
    ## Set up align_dataset object
    align_data_location = '/g/ghez/abhimat/datasets/align_data/'
    align_name = 'phot_19_04_Kp_kp'

    align_pickle_loc = align_data_location + 'alignPickle_' + align_name + '.pkl'

    align_data = align_dataset.align_dataset(align_pickle_loc)
    
    ## Read in stored variables from align_dataset object
    epoch_dates = align_data.epoch_dates
    epoch_MJDs = align_data.epoch_MJDs
    star_names = align_data.star_names

    star_mags = align_data.star_mags_neighCorr
    star_magErrors = align_data.star_magErrors_neighCorr
    star_magMeans = align_data.star_magMeans_neighCorr
    
    ## Tests
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as font_manager
    from matplotlib.ticker import MultipleLocator
    
    ### Plot Nerdery
    plt.rc('font', family='serif')
    plt.rc('font', serif='Computer Modern Roman')
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r"\usepackage{gensymb}")
    
    plt.rc('xtick', direction = 'in')
    plt.rc('ytick', direction = 'in')
    plt.rc('xtick', top = True)
    plt.rc('ytick', right = True)
    
    
    for cur_star in ['S4-258', 'irs16C', 'irs16SW', 'S2-36']:
        fig = plt.figure(figsize=(7,4))
        
        ## Perform fit
        (mags_polyfit, model, poly_coeffs, dof, red_chiSq) = lc_polyfit(star_mags[cur_star],
                                                                        star_magErrors[cur_star],
                                                                        epoch_MJDs)
        
        detected_filter = np.where(star_mags[cur_star] > 0.0)
        mags_det = (star_mags[cur_star])[detected_filter]
        mag_errs_det = (star_magErrors[cur_star])[detected_filter]
        obs_times_det = epoch_dates[detected_filter]
    
        num_obs = len(mags_det)
        
        
        ## Print output
        print('{0}'.format(cur_star))
        print((mags_polyfit, model, poly_coeffs, dof, red_chiSq))
        print('')
        
        
        ## Draw plot
        fig = plt.figure(figsize=(7, 5))
        
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.errorbar(obs_times_det, mags_det, yerr=mag_errs_det, fmt='k.')
        ax1.plot(obs_times_det, model, 'r-')
        
        ax1.invert_yaxis()
        
        x_majorLocator = MultipleLocator(2.)
        x_minorLocator = MultipleLocator(0.5)
        ax1.xaxis.set_major_locator(x_majorLocator)
        ax1.xaxis.set_minor_locator(x_minorLocator)

        y_majorLocator = MultipleLocator(0.2)
        y_minorLocator = MultipleLocator(0.05)
        ax1.yaxis.set_major_locator(y_majorLocator)
        ax1.yaxis.set_minor_locator(y_minorLocator)
        
        ax1.set_title(r"{0}".format(cur_star.replace('irs', 'IRS ')))
        ax1.set_ylabel(r"$m_{K'}$")
        
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.errorbar(obs_times_det, mags_polyfit, yerr=mag_errs_det, fmt='k.')
        ax2.plot(obs_times_det, model*0 + star_magMeans[cur_star], 'r-')
        
        ax2.invert_yaxis()
        
        x_majorLocator = MultipleLocator(2.)
        x_minorLocator = MultipleLocator(0.5)
        ax2.xaxis.set_major_locator(x_majorLocator)
        ax2.xaxis.set_minor_locator(x_minorLocator)

        y_majorLocator = MultipleLocator(0.2)
        y_minorLocator = MultipleLocator(0.05)
        ax2.yaxis.set_major_locator(y_majorLocator)
        ax2.yaxis.set_minor_locator(y_minorLocator)
        
        ax2.set_ylabel(r"m_{K'}")
        ax2.set_xlabel('Observation Time')
        
        fig.tight_layout()
        fig.savefig('./{0}.pdf'.format(cur_star))
        plt.close(fig)
        
    