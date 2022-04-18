import numpy as np

####################################################################################################
### This module contains a single function that calculates the total, county-level VCP emissions speciated and ordered.
####################################################################################################

####################################################################################################
def speciated_allocation(year,chem_index,county_fips):

    ### Import speciated emissions.
    spec_emis    = np.genfromtxt('./output/emissions_by_subpuc/'+str(year)+'/speciated_emissions_by_subpuc_'+str(year)+'.csv',skip_header=1,delimiter=',')                               # speciation by sub-PUC
    spec_emis    = spec_emis[:] / np.sum(spec_emis[:,:],axis=0)

    ### Import sub-PUC level emissions at County-level
    emissions    = np.genfromtxt('./output/emissions_spatially_allocated/'+str(year)+'/subpuc_county_TOG_emissions_'+str(year)+'.csv',delimiter=',',skip_header=2)  # sub-PUC level, total emissions [kg/yr]
    emissions    = emissions[:,2:]

    for i in range(len(county_fips)):
        final_array  = spec_emis[:,:] * emissions[i,:]
        final_array  = np.nansum(final_array[:,:],axis=1)
        index        = np.argsort(-final_array[:])
        final_array  = final_array[index]
    
        for j in range(len(final_array)):
            if final_array[j] < 1e-6:
                cutoff = j
                break
            else: pass

        index  = index[0:cutoff]

        ############################################################################################
        headerline    = 'Chemical Name,Emissions [kg/yr]'
        output_file   = './output/speciated_emissions_spatially_allocated/'+str(year)+'/'+county_fips[i]+'_'+str(year)+'.csv'
        np.savetxt(output_file,np.column_stack((chem_index[index],final_array[0:cutoff])),delimiter=',',fmt='%s',header=headerline)
        ############################################################################################

####################################################################################################