import numpy as np

####################################################################################################
### This module contains a single function that calculates the SOA and O3 potential for all states and counties
####################################################################################################

####################################################################################################
def aq_potential(year,subpuc_names):

    ################################################################################################
    ### Import State and County emissions. 
    state_emissions  = np.genfromtxt('./output/emissions_spatially_allocated/'+str(year)+'/subpuc_state_TOG_emissions_'+str(year)+'.csv',delimiter=',',skip_header=2)   # fipstate, fipcty, emissions
    county_emissions = np.genfromtxt('./output/emissions_spatially_allocated/'+str(year)+'/subpuc_county_TOG_emissions_'+str(year)+'.csv',delimiter=',',skip_header=2)  # fipstate, fipcty, emissions
    ### Import sub-PUC specific effective SOA Yields and MIR values.
    subPUC_effective = np.genfromtxt('./output/emissions_by_subpuc/'+str(year)+'/summary_by_subpuc_'+str(year)+'.csv',delimiter=',',skip_header=1,usecols=(15,16))      # SOAYield.%, MIR.gO3/g
    ################################################################################################
    
    ################################################################################################
    ### Initialize the final arrays. 
    ### Final SOA potential, State-level [Gg/year]
    final_state_SOA_array         = np.zeros((len(state_emissions),len(subpuc_names)+2))
    final_state_SOA_array[:,0:2]  = state_emissions[:,0:2]
    ### Final SOA potential, County-level [kg/year]
    final_county_SOA_array        = np.zeros((len(county_emissions),len(subpuc_names)+2))
    final_county_SOA_array[:,0:2] = county_emissions[:,0:2]

    ### Final O3 potential, State-level [Gg/year]
    final_state_O3_array          = np.zeros((len(state_emissions),len(subpuc_names)+2))
    final_state_O3_array[:,0:2]   = state_emissions[:,0:2]
    ### Final O3 potential, County-level [kg/year]
    final_county_O3_array         = np.zeros((len(county_emissions),len(subpuc_names)+2))
    final_county_O3_array[:,0:2]  = county_emissions[:,0:2]
    ################################################################################################

    ################################################################################################
    ### Calculate State-level and County-level SOA and O3 potential for each sub-PUC
    for i in range(len(subPUC_effective)):
        final_state_SOA_array[:,2+i]  = state_emissions[:,2+i]  * subPUC_effective[i,0] / 100
        final_county_SOA_array[:,2+i] = county_emissions[:,2+i] * subPUC_effective[i,0] / 100
        final_state_O3_array[:,2+i]   = state_emissions[:,2+i]  * subPUC_effective[i,1]
        final_county_O3_array[:,2+i]  = county_emissions[:,2+i] * subPUC_effective[i,1]
    ################################################################################################

    ################################################################################################
    ### 
    headerline1   = 'fipstate,fipscty,'+np.array2string(subpuc_names[:],max_line_width=1e6,separator=',') 
    headerline2   = 'All SOA potential reported in Gg/year'
    headerline    = '\n'.join([headerline1,headerline2])
    output_file   = './output/air_quality_potential/'+str(year)+'/subpuc_state_SOA_potential_'+str(year)+'.csv'
    np.savetxt(output_file,final_state_SOA_array[:],delimiter=',',header=headerline)
    ################################################################################################

    ################################################################################################
    ###
    headerline1   = 'fipstate,fipscty,'+np.array2string(subpuc_names[:],max_line_width=1e6,separator=',') 
    headerline2   = 'All SOA potential reported in kg/year'
    headerline    = '\n'.join([headerline1,headerline2])
    output_file   = './output/air_quality_potential/'+str(year)+'/subpuc_county_SOA_potential_'+str(year)+'.csv'
    np.savetxt(output_file,final_county_SOA_array[:],delimiter=',',header=headerline)
    ################################################################################################

    ################################################################################################
    ### 
    headerline1   = 'fipstate,fipscty,'+np.array2string(subpuc_names[:],max_line_width=1e6,separator=',') 
    headerline2   = 'All O3 potential reported in Gg/year'
    headerline    = '\n'.join([headerline1,headerline2])
    output_file   = './output/air_quality_potential/'+str(year)+'/subpuc_state_O3_potential_'+str(year)+'.csv'
    np.savetxt(output_file,final_state_O3_array[:],delimiter=',',header=headerline)
    ################################################################################################

    ################################################################################################
    ###
    headerline1   = 'fipstate,fipscty,'+np.array2string(subpuc_names[:],max_line_width=1e6,separator=',')
    headerline2   = 'All O3 potential reported in kg/year'
    headerline    = '\n'.join([headerline1,headerline2])
    output_file   = './output/air_quality_potential/'+str(year)+'/subpuc_county_O3_potential_'+str(year)+'.csv'
    np.savetxt(output_file,final_county_O3_array[:],delimiter=',',header=headerline)
    ################################################################################################

####################################################################################################