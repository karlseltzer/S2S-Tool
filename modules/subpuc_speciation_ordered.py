import numpy as np

####################################################################################################
### These functions generate csv files for each sub-PUC ordered by species emissions.
####################################################################################################

####################################################################################################
def oxycar_ratio(chem_props_vars):
    oc_ratio = chem_props_vars[:,2] / chem_props_vars[:,1]
    oc_ratio[np.isnan(oc_ratio[:])] = 0.0
    return oc_ratio
####################################################################################################

####################################################################################################
def order_subpucs(year,subpuc_names,chem_index,chem_props_vars,chem_props_strs,oc_ratio):
    
    roc_emis = np.genfromtxt('./output/emissions_by_subpuc/'+str(year)+'/speciated_emissions_by_subpuc_'+str(year)+'.csv',delimiter=",",skip_header=1)

    for i in range(len(subpuc_names)):
        final_array      = np.zeros((len(roc_emis),6)) # O:C, log(C*), SOA Yield, MIR, Emissions [kg/person/year], % of Emissions
        index            = np.argsort(-roc_emis[:,i])
        final_array[:,0] = oc_ratio[index]
        final_array[:,1] = chem_props_vars[index,5]
        final_array[:,2] = chem_props_vars[index,6]
        final_array[:,3] = chem_props_vars[index,7]
        final_array[:,4] = roc_emis[index,i]
        final_array[:,5] = roc_emis[index,i] / np.sum(roc_emis[:,i]) * 100

        for j in range(len(final_array)):
            if final_array[j,5] < 1e-6: 
                cutoff = j
                break
            else: pass

        index            = index[0:cutoff]

        ############################################################################################
        ### Generate a per sub-PUC summary output file.
        headerline1   = 'Chemical Name,Group,HAP,SPECIATE_ID,O:C,log(C*),SOA Yield,MIR,Emissions [kg/person/year],% of Emissions'
        output_file   = './output/emissions_by_subpuc/'+str(year)+'/'+subpuc_names[i]+'_summary_'+str(year)+'.csv'
        np.savetxt(output_file,np.column_stack((chem_index[index],chem_props_strs[index,0:2],chem_props_vars[index,0],final_array[0:cutoff,:])),delimiter=',',fmt='%s',header=headerline1)
####################################################################################################

####################################################################################################
def order_total(year,chem_index,chem_props_vars,chem_props_strs,oc_ratio):
    
    roc_emis = np.genfromtxt('./output/emissions_by_subpuc/'+str(year)+'/speciated_emissions_by_subpuc_'+str(year)+'.csv',delimiter=",",skip_header=1)
    roc_emis = np.nansum(roc_emis[:,:],axis=1)

    final_array      = np.zeros((len(roc_emis),6)) # O:C, log(C*), SOA Yield, MIR, Emissions [kg/person/year], % of Emissions
    index            = np.argsort(-roc_emis[:])
    final_array[:,0] = oc_ratio[index]
    final_array[:,1] = chem_props_vars[index,5]
    final_array[:,2] = chem_props_vars[index,6]
    final_array[:,3] = chem_props_vars[index,7]
    final_array[:,4] = roc_emis[index]
    final_array[:,5] = roc_emis[index] / np.sum(roc_emis[:]) * 100

    for j in range(len(final_array)):
        if final_array[j,5] < 1e-6: 
            cutoff = j
            break
        else: pass

    index            = index[0:cutoff]

    ################################################################################################
    ### Generate a total summary output file.
    headerline1   = 'Chemical Name,Group,HAP,SPECIATE_ID,O:C,log(C*),SOA Yield,MIR,Emissions [kg/person/year],% of Emissions'
    output_file   = './output/emissions_by_subpuc/'+str(year)+'/TOTAL_summary_'+str(year)+'.csv'
    np.savetxt(output_file,np.column_stack((chem_index[index],chem_props_strs[index,0:2],chem_props_vars[index,0],final_array[0:cutoff,:])),delimiter=',',fmt='%s',header=headerline1)
####################################################################################################