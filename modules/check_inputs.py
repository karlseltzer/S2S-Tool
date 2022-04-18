import sys
import numpy as np

####################################################################################################
### These functions all perform QA checks on input files. 
### These should catch many errors, but is not exhaustive. 
####################################################################################################

####################################################################################################
def check_usage(subpuc_names,year,subpuc_usage):
    if len(subpuc_names) == len(subpuc_usage[1:]):
        pass
    else: sys.exit('There is an issue with your subpuc_usage.csv file. Number of sub-PUC(s) is incorrect.')
    year_available = 0
    for i in range(len(subpuc_usage[0,1:])):
        if subpuc_usage[0,1+i] == year:
            year_available = 1
        else: pass
    if year_available == 0:
        sys.exit('There is an issue with your subpuc_usage.csv file. '+str(year)+' is missing.')
    else: pass
####################################################################################################

####################################################################################################
def check_usetime(subpuc_names,subpuc_usetime):
    if len(subpuc_names) == len(subpuc_usetime):
        pass
    else: sys.exit('There is an issue with your subpuc_usetimescales.csv file. Number of sub-PUC(s) is incorrect.')
    for i in range(len(subpuc_usetime)):
        if subpuc_usetime[i,1] >= 0.0 and subpuc_usetime[i,1] <= 6.0:
            pass
        else: sys.exit('There is a bounds issue in your subpuc_usetimescales.csv files.')
####################################################################################################

####################################################################################################
def check_controls(subpuc_names,subpuc_controls):
    if len(subpuc_names) == len(subpuc_controls):
        pass
    else: sys.exit('There is an issue with your subpuc_controls.csv file. Number of sub-PUC(s) is incorrect.')
    for i in range(len(subpuc_controls)):
        if subpuc_controls[i,1] >= 0.0 and subpuc_controls[i,1] <= 1.0:
            pass
        else: sys.exit('There is a bounds issue in your subpuc_controls.csv files.')
####################################################################################################

####################################################################################################
def check_1st_order_spec(subpuc_names,first_ord_spec):
    if len(subpuc_names) == len(first_ord_spec):
        pass
    else: sys.exit('There is an issue with your subpuc_1st_order_speciation.csv file. Number of sub-PUC(s) is incorrect.')
    for i in range(len(first_ord_spec)):
        if np.sum(first_ord_spec[i,0:3]) >= 0.99 and np.sum(first_ord_spec[i,0:3]) <= 1.01:
            pass
        else: sys.exit('There is an issue with your subpuc_1st_order_speciation.csv file. Water + Inorganic + Organic out of bounds.')
        if first_ord_spec[i,2] >= first_ord_spec[i,3]:
            pass
        else: sys.exit('There is an issue with your subpuc_1st_order_speciation.csv file. TOG > Organic.')
        for j in range(len(first_ord_spec[0,:])):
            if first_ord_spec[i,j] >= 0.0 and first_ord_spec[i,j] <= 1.0:
                pass
            else: sys.exit('There is a bounds issue in your subpuc_1st_order_speciation.csv files.')
####################################################################################################

####################################################################################################
def check_organic_spec(subpuc_names,organic_spec,chem_index):
    if len(subpuc_names) == len(organic_spec[0,:]):
        pass
    else: sys.exit('There is an issue with your subpuc_organic_speciation.csv file. Number of sub-PUC(s) is incorrect.')
    for i in range(len(organic_spec[0,:])):
        if np.nansum(organic_spec[1:,i]) >= 0.99 and np.nansum(organic_spec[1:,i]) <= 1.01:
            pass
        else: sys.exit('There is an issue with your subpuc_organic_speciation.csv file. Total speciation out of bounds.')
    if len(chem_index) == len(organic_spec[1:,0]):
        pass
    else: sys.exit('There is an issue with your subpuc_organic_speciation.csv file. Number of species is incorrect.')
####################################################################################################

####################################################################################################
def check_chem_assignments(chem_props_vars,chem_props_strs,chem_index):
    if len(chem_index) == len(chem_props_vars) and len(chem_index) == len(chem_props_strs):
        pass
    else: sys.exit('There is an issue with your chemical_assignments.csv file. Number of species is incorrect.')
####################################################################################################
    
####################################################################################################
def check_subpuc_SCC_map(subpuc_scc_map,subpuc_names):
    if len(subpuc_names) == len(subpuc_scc_map):
        pass
    else: sys.exit('There is an issue with your subpuc_SCC_map.csv file. Number of sub-PUC(s) is incorrect.')
    if np.array_equal(subpuc_names[:],subpuc_scc_map[:,1]):
        pass
    else: sys.exit('There is an issue with your subpuc_SCC_map.csv file. sub-PUCs in wrong order.')
####################################################################################################