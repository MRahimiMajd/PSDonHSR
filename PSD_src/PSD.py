import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import r2_score

###############################################################################
def find_exponents(spectrum_arr, permutation=False):
    """
    This function finds the low- and high-frequency characteristic frequency 
    domains of the given HSR spectrum and returns the power-law regression 
    statistics for these domains as well as for the entire frequency domain.
    
    Parameters
    ----------
    spectrum_arr: array_like
        Time series of reflectance values in increasing wavelength order with a
        fixed sampling frequency of 1.0/1nm.
        Note: 
            If the data has a different fixed sampling frequency, the 'fs' parameter
            in the 'scipy.signal.periodogram()' function must be modified accordingly. 
    permutation: bool, optional
        If True, permutes the reflectance values before analysis. The default is False.
    Returns 
    -------
    psd_stats : list 
        The resulting statistics, including:
            alpha_l: The power-law exponent of the low-frequency range.
            
            ks_l: The  Kolmogorov-Smirnov distance (KS_D) between the empirical
            and fitted PSD values of the low-frequency range.
            
            R2_l: The coefficient of determination (R2) of the linear fit of the
            low-frequency domain on the logarithmic domain.
            
            alpha_h: The power-law exponent of the high-frequency domain.
            
            ks_h: The  Kolmogorov-Smirnov distance (KS_D) between the empirical 
            and fitted PSD values of the high-frequency domain.
            
            R2_h: The coefficient of determination (R2) of the linear fit of the
            high-frequency domain on the logarithmic scale.
            
            alpha_t: The power-law exponent of the entire frequency domain.
            
            ks_t: The  Kolmogorov-Smirnov distance (KS_D) between the empirical 
            and fitted PSD values of the entire frequency domain.
            
            R2_t: The coefficient of determination (R2) of the linear fit of the
            entire frequency domain on the logarithmic scale.
    """
    # Define the spectrum and permute it if required.
    spectrum=np.array(spectrum_arr)
    if permutation: spectrum = np.random.permutation(spectrum)
    
    # Calculate the power spectral density and the corresponding Fourier 
    # frequencies of the spectrum.
    freq, PSD=scipy.signal.periodogram(spectrum, fs=1.0, window='boxcar', 
                                       nfft=None, detrend='constant', 
                                       return_onesided=True, scaling='density',
                                       axis=-1)
    
    # Exclude the zero frequency (f=0) and its corresponding density value from 
    # the 'PSD' and 'freq' arrays.
    freq=freq[1:]
    PSD=PSD[1:]
    
    # Define the 'PSD' and 'freq' arrays in Log10 scale.
    freq_log=np.log10(freq)
    PSD_log=np.log10(PSD)
    ###########################################################################
    """ Calculate the power-law statistics for the entire frequency domain:"""
    x_t=np.array(freq_log)
    y_t=np.array(PSD_log)
    reg_results_t = scipy.stats.linregress(x_t, y_t)
    slope_t, intercept_t = reg_results_t.slope, reg_results_t.intercept
    y_pred_t = (slope_t * x_t) + intercept_t
    R2_t=r2_score(y_t,y_pred_t)

    # Find the effective cumulative distributions of PSD and its power-law fit in
    # the selected entire frequency domain by normalizing them, to sum up to 1, on the
    # main (non-logarithmic) scale. 
    cdf1=np.cumsum(np.power(10,y_t)/np.sum(np.power(10,y_t))) 
    cdf2=np.cumsum(np.power(10,y_pred_t)/np.sum(np.power(10,y_pred_t))) 
    # Find the KS distance between 'y_t' and 'y_pred_t' for the entire frequency domain.
    ks_t=np.amax(np.abs(cdf1-cdf2)) 
    ###########################################################################
    """ Define the low-frequency domain and calculate its power-law statistics:"""
    start_index_l=0 # starting index of low-frequency domain.

    # Define the minimum frequency point to which the low-frequency range can be
    # extended from start_index_l.
    freq_f1_l=freq_log[0]+1.0
    index_f1_l = np.searchsorted(freq_log, freq_f1_l, side='right')
    # Define the maximum frequency point to which the low-frequency range can be
    # extended from start_index_l.
    freq_f2_l=freq_log[-1]-1.0
    index_f2_l = np.searchsorted(freq_log, freq_f2_l, side='right')
    ##############################
    # determenation of the low-frequency range by iteration over the cut-off 
    # frequencies with the starting frequency fixed at f=0 to find the best cut-off.
    KSs_l={}
    for index_f in range(index_f1_l,index_f2_l):
        x=np.array(freq_log[start_index_l:index_f])
        y=np.array(PSD_log[start_index_l:index_f])
        reg_results = scipy.stats.linregress(x, y)
        y_pred = (reg_results.slope * x) + reg_results.intercept
        cdf1=np.cumsum(np.power(10,y)/np.sum(np.power(10,y)))
        cdf2=np.cumsum(np.power(10,y_pred)/np.sum(np.power(10,y_pred)))
        ks=np.amax(np.abs(cdf1-cdf2))
        KSs_l[index_f-1]=ks
        
    cut_off_index_l = min(KSs_l, key=KSs_l.get)
    ###############################
    x_l=np.array(freq_log[start_index_l:cut_off_index_l+1])
    y_l=np.array(PSD_log[start_index_l:cut_off_index_l+1])
    reg_results_l = scipy.stats.linregress(x_l, y_l)
    slope_l, intercept_l = reg_results_l.slope, reg_results_l.intercept
    y_pred_l = (slope_l * x_l) + intercept_l
    R2_l=r2_score(y_l,y_pred_l)

    # Find the effective cumulative distributions of PSD and its power-law fit in
    # the selected low-frequency domain by normalizing them, to sum up to 1, on the
    # main (non-logarithmic) scale. 
    cdf1=np.cumsum(np.power(10,y_l)/np.sum(np.power(10,y_l)))
    cdf2=np.cumsum(np.power(10,y_pred_l)/np.sum(np.power(10,y_pred_l)))
    # Find the KS distance between 'y_t' and 'y_pred_t' for the low-frequency domain.
    ks_l=np.amax(np.abs(cdf1-cdf2))
    ###########################################################################
    """ Define the high-frequency domain and calculate its power-law statistics:"""
    start_index_h = cut_off_index_l # starting index of high-frequency domain.
    
    # Define the minimum frequency point to which the high-frequency range can be
    # extended from start_index_h.
    freq_f1_h=freq_log[start_index_h]+ 1.0 
    index_f1_h = np.searchsorted(freq_log, freq_f1_h, side='right')
    # The maximum frequency index to which the high-frequency range can be
    # extended from start_index_h.
    index_f2_h = len(freq_log)
    ###############################
    # determenation of the high-frequency range by iteration over the cut-off 
    # frequencies with the starting frequency fixed at cutt_off_index_l to find the best cut-off.
    KSs_h={}
    for index_f in range(index_f1_h,index_f2_h):
        x=np.array(freq_log[start_index_h:index_f])
        y=np.array(PSD_log[start_index_h:index_f])
        reg_results = scipy.stats.linregress(x, y)
        y_pred = (reg_results.slope * x) + reg_results.intercept
        cdf1=np.cumsum(np.power(10,y)/np.sum(np.power(10,y)))
        cdf2=np.cumsum(np.power(10,y_pred)/np.sum(np.power(10,y_pred)))
        ks=np.amax(np.abs(cdf1-cdf2))
        KSs_h[index_f-1]=ks
    cut_off_index_h = min(KSs_h, key=KSs_h.get)
    ###############################
    x_h=np.array(freq_log[start_index_h:cut_off_index_h+1])
    y_h=np.array(PSD_log[start_index_h:cut_off_index_h+1])
    reg_results_h = scipy.stats.linregress(x_h, y_h)
    slope_h, intercept_h = reg_results_h.slope, reg_results_h.intercept
    y_pred_h = (slope_h * x_h) + intercept_h
    R2_h=r2_score(y_h,y_pred_h)
    
    # Find the effective cumulative distributions of PSD and its power-law fit in
    # the selected high-frequency domain by normalizing them, to sum up to 1, on the
    # main (non-logarithmic) scale. 
    cdf1=np.cumsum(np.power(10,y_h)/np.sum(np.power(10,y_h)))
    cdf2=np.cumsum(np.power(10,y_pred_h)/np.sum(np.power(10,y_pred_h)))
    # Find the KS distance between 'y_t' and 'y_pred_t' for the high-frequency domain.
    ks_h=np.amax(np.abs(cdf1-cdf2))
    ###############################
    alpha_l, alpha_h, alpha_t = abs(slope_l), abs(slope_h), abs(slope_t)
    psd_stats= [alpha_l, ks_l, R2_l, alpha_h ,ks_h, R2_h, alpha_t, ks_t, R2_t]
    
    return psd_stats
###############################################################################
def psd_analysis(data_df, permute=False, first_lambda=350, start_lambda=400): 
    """
    This function determines the chrastersitic frequency ranges and computes the
    PSD statistics of them for each of the HSR samples of the given data set.

    Parameters
    ----------
    data_df : Pandas DataFrame
        The Pandas DataFrame where data for each sample line of the given species
        is provided in a separate row. 
        The data includes: i) columns representing information about the line, 
        ii) columns representing the measured parameters for the line, and iii)
        columns showing the hyperspectral reflectance (HSR) of the line.
        
        Note: 
            - All the information about the lines, i.e., ID, measurement details, 
            and parameter values are expected to be located in the columns before the
            (left side of) those representing the HSR values for different wavelengths.
            - The HSR data for different wavelengths (as different columns) is
            necessarily expected to be in the increasing order from left to right.
            - The names of the columns representing the HSR data are expected to 
            be as the string format of the corresponding wavelength values.
    permute : bool, optional
        If True, the HSR data will be permuted across wavelengths, separately for
        each sample, before analysis. The default is False.
    first_lambda : float, optional,
        The minimum wavelength of the HSR data. The default is 350.
    start_lambda : float, optional,
        The minimum wavelength of the HSR data required to be analyzed. The default is 400.
        Note:
            The start_lambda must be greater than or equal to the first_lambda. 

    Returns
    -------
    res_df: Pandas DataFrame
       The results of PSD analyses in different frequency domains for all the lines
       together with the information and measured parameters of them.
    """
    data=data_df.copy(deep=True)
    
    first_column_index=list(data.keys()).index(str(first_lambda)) 
    start_column_index=list(data.keys()).index(str(start_lambda))
    info_cols = data.keys()[:first_column_index].tolist() # The columns of information and parameters.    
    res_cols=info_cols+['alpha_l','KS_l','R2_l','alpha_h','KS_h','R2_h','alpha_t','KS_t','R2_t']
    Res=[]
    for i in range(len(data)):
        spectrum_HSR=data.iloc[i,start_column_index:].to_numpy(dtype='float64',copy=True)
        print('HSR sample number:',i)
        res_psd=find_exponents(spectrum_HSR, permutation=permute)
        info=data.iloc[i,:first_column_index].tolist()
        res_row=info + res_psd
        Res.append(res_row)
    res_df=pd.DataFrame(Res,columns=res_cols)
    return res_df
###############################################################################
def dataset_psd_results(dataset_fname, res_fname,permut_HSR=False, first_lambda=350, start_lambda=400):
    '''
    This function reads the leaf hyperspectral reflectance (HSR) dataset of a given 
    species from one file and saves the results of its power spectral density (PSD) 
    analysis to another file.
    
    Parameters
    ----------
    dataset_fname :str
        Path to the file of HSR dataset in the '.csv' format.
        Note: 
            - The rows must contain the data for individual samples. 
            - The columns must contain HSR values for different wavelengths 
            (with a fixed bin size) in ascending order. The HSR columns must be
            named as the string format of the wavelengths (e.g., '400'), and the values 
            of these columns must be in a numeric format. 
            - In addition to the HSR columns, the dataset may contain any information 
            about the sample line (in numeric or str format), and the columns 
            representing the measured values for that line. These columns must be
            before (in the left side of) the HSR columns. The resulting file will
            contain all the columns except the HSR columns.
    res_fname :str
        Path to the results file (for saving).
    permut_HSR : bool, optional
        If True, the HSR data will be permuted across wavelengths, separately for
    first_lambda : float, optional,
        The minimum wavelength of the HSR data. The default is 350.
    start_lambda : float, optional,
        The minimum wavelength of the HSR data required to be analyzed. The default is 400.
        Note:
            The start_lambda must be greater than or equal to the first_lambda. 
        
    Returns
    ------- 
    None.

    '''
    detaset=pd.read_csv(dataset_fname) 
    result_df=psd_analysis(detaset, permute=permut_HSR, first_lambda=first_lambda, start_lambda=start_lambda)
    result_df.to_csv(res_fname,index=False)
###############################################################################
