import numpy as np
import pandas as pd
from scipy import stats
import os
###############################################################################
"""
Note: The following functions are specific for the post-processing of the results 
for CAM Maize 2021 and CAM Maize 2022 based on their measured traits of, their 
estimated traits by PROSPECT D inversion, and the results of their PSD analysis. 
These functions aim to regenerate the results presented in the paper. 
The code lines of the functions must be adopted for use on other data sets, accordingly. 

list_of_ species_names = ['CAM_Maize_2021', 'CAM_Maize_2022']   

"""
###############################################################################
###############################################################################
def correlation_CAM_Maize(species_name, r_method = 'Pearson' ,with_inversion=True):
    '''
    Calculate the correlations between PSD exponents and their corresponding leaf
    traits in the data sets CAM Maize 2021 and CAM Maize 2022. 
    
    Parameters
    ----------
    species_name : str
        Name of the species (data set) from CAM Maize 2021 and CAM Maize 2022.
    r_method: 
        The correlation method. This parameter must be 'Pearson' (linear) or 'Spearman' (non-linear). The default is 'Pearson'.
    with_inversion : bool, optional
        If True, the six traits estimated by PROSPECT inversion model are included in the analyses. The default is True.

    Returns
    -------
    r_res : Pandas DataFrame.
        The correlation results.

    '''
    ######################################
    script_dir = '' # Must be specified according to the location of the files.
    file_name = f'{species_name}_PSD_results.csv' # Must be specified according to the filename of the PSD results.
    res=pd.read_csv(os.path.join(script_dir, file_name))
    #######################################
    Traits_dict={'CAM_Maize_2021':['V_pmax','V_max' ,'S.Lim.','A_400', 'gs_w','iWUE','SLA'
                                    , 'C_i', 'Nit','A_sat', 'gs_sat', 'C_i_sat', 'iWUE_sat'],
              'CAM_Maize_2022':['A_sat', 'gs_sat', 'C_i_sat', 'iWUE_sat','SLA','W_leaf']}
    traits=Traits_dict[species_name]
    inversion_traits=['CHL_inv', 'CAR_inv', 'EWT_inv', 'LMA_inv', 'ANT_inv','N_inv']
    #######################################
    if with_inversion:
        inversion_file_name=f'{species_name}_inversion_param.csv'
        res_inversion=pd.read_csv(os.path.join(script_dir,inversion_file_name))
        res_inversion=res_inversion[['ID','N','CHL','CAR','EWT','LMA','ANT']]
        res_inversion.columns=['ID',r'N_inv',r'CHL_inv',r'CAR_inv',r'EWT_inv',r'LMA_inv','ANT_inv']
        res=pd.merge(res,res_inversion, on='ID',how='right')
        traits=inversion_traits+traits
    corrs=[]
    for trait in traits:
        df=res[['alpha_l','alpha_h','alpha_t',trait]].dropna().copy(deep=True)
        if r_method == 'Pearson':
            r_l=stats.pearsonr(df['alpha_l'],df[trait])
            r_h=stats.pearsonr(df['alpha_h'],df[trait])
            r_t=stats.pearsonr(df['alpha_t'],df[trait])
        elif r_method == 'Spearman':
            r_l=stats.spearmanr(df['alpha_l'],df[trait])
            r_h=stats.spearmanr(df['alpha_h'],df[trait])
            r_t=stats.spearmanr(df['alpha_t'],df[trait])
        else:
            raise ValueError(" 'r_method' must be 'Pearson' or 'Spearman'")
                
        corrs.append([df.shape[0], r_l[0],r_l[1],r_h[0],r_h[1],r_t[0],r_t[1]]) 
    
    r_res=pd.DataFrame(np.array(corrs),index=traits,columns=['N','r_alpha_l','pval_alpha_l'
                              ,'r_alpha_h','pval_alpha_h','r_alpha_t','pval_alpha_t'])
    return r_res
###############################################################################
def correlation_between_rs(species_name,with_inversion=True):
    '''
    Calculate the linear correlation between r_alpha_l and r_alpha_h across the given traits.
    
    Parameters
    ----------
    species_name : str
        Name of the species (data set) from CAM Maize 2021 and CAM Maize 2022.
    with_inversion : bool, optional
        If True, the six traits estimated by PROSPECT inversion model are included in the analyses. The default is True.

    Returns
    -------
    r_res : dict
        The linear correlation between r_alpha_l and r_alpha_h.

    '''
    corr_df=correlation_CAM_Maize(species_name,with_inversion=with_inversion)
    rl_rh=stats.pearsonr(corr_df['r_alpha_l'],corr_df['r_alpha_h'])
    print('The linear corraletion between r_alpha_l and r_alpha_h is:',rl_rh[0]) 
    return {'correlation':rl_rh[0],'p-value':rl_rh[1]}
###############################################################################
def corr_between_years():
    '''
    Calculate the The linear correlations between the average of the exponents 
    alpha_l, alpha_h, and alpha_t over replicates of the shared maize lines of 
    CAM Maize 2021 and CAM Maize 2022 data sets.
    
    Returns
    -------
    rs : dict
        The correlation results between the shared lines.

    '''
    species_names=['CAM_Maize_2021', 'CAM_Maize_2022']   
    script_dir = '' # Must be specified according to the location of the files.
    file_name_maize_21 = f'{species_names[0]}_PSD_results.csv' # Must be specified according to the filename of the PSD results for CAM_Maize_2021.
    file_name_maize_22 = f'{species_names[1]}_PSD_results.csv' # Must be specified according to the filename of the PSD results for CAM_Maize_2022.
    
    exp_maize_21=pd.read_csv(os.path.join(script_dir, file_name_maize_21))
    exp_maize_21=exp_maize_21[['Accession','alpha_l','alpha_h','alpha_t']]
    exp_maize_21 = exp_maize_21.rename(columns={'alpha_l':'alpha_l_2021', 'alpha_h':'alpha_h_2021', 'alpha_t':'alpha_t_2021'})
    mean_21 = exp_maize_21.groupby('Accession').mean()
    
    exp_maize_22=pd.read_csv(os.path.join(script_dir, file_name_maize_22))
    exp_maize_22=exp_maize_22[['Accession','alpha_l','alpha_h','alpha_t']]
    exp_maize_22 = exp_maize_22.rename(columns={'alpha_l':'alpha_l_2022', 'alpha_h':'alpha_h_2022', 'alpha_t':'alpha_t_2022'})
    mean_22 = exp_maize_22.groupby('Accession').mean()
    
    mean=pd.merge(mean_21,mean_22,on='Accession',how='inner')
    Exps=['alpha_l','alpha_h','alpha_t'] 
    
    rs={}
    for i,exp in enumerate(Exps):
        rs[exp]=stats.pearsonr(mean[exp+'_2021'],mean[exp+'_2022'])
    return rs
###############################################################################
def alpha_variences(species_name):
    '''
    Compute the variances of the exponent values (alpha_l, alpha_h, and alpha_t)
    over replicates of different lines of CAM Maize 2021, and CAM Maize 2022 data sets.

    Parameters
    ----------
    species_name : str
        Name of the species (data set) from CAM Maize 2021 and CAM Maize 2022.

    Returns
    -------
    variances : Pandas DataFrame
        The variances of the exponent values over the replicates of the lines.

    '''
    script_dir = '' # Must be specified according to the location of the files.
    file_name = f'{species_name}_PSD_results.csv' # Must be specified according to the filename of the PSD results.
    res=pd.read_csv(os.path.join(script_dir, file_name)) 
    
    
    exp_df=res[['Accession','alpha_l','alpha_h','alpha_t']]
    
    variances = exp_df.groupby('Accession').var()
    variances = variances.dropna()

    return variances
###############################################################################

