import sys
import numpy as np
from datetime import date,datetime
import pandas as pd

####################################################################################################
### This function generates a gscnv file for the target MECHANISM.
####################################################################################################

####################################################################################################
def append_molwght(molecular_wghts,mech4import):
    
    final_array        = []
    
    for i in range(len(mech4import)):
        target_spec    = molecular_wghts[molecular_wghts[:,1]==mech4import[i,2]]
        final_array    = np.append(final_array,target_spec[0,2])
    
    return(final_array)

####################################################################################################
def gen_gspro(profiles,species,species_props,molecular_wghts,mech4import,MECHANISM):
    
    ### gscnv file columns
    column_names  = ['PROFILE','INPUT.POLL','MODEL.SPECIES','MASS.FRACTION','MOLECULAR.WGHT']
    dfgspro       = pd.DataFrame(columns=column_names)

    for i in range(len(profiles)):
        prof        = profiles[i,0]
        i_poll      = 'TOG'
        target_spec = species[species[:,0]==prof]
        for j in range(len(target_spec)):
            spec_fraction    = np.float(target_spec[j,2]) / 100
            target_modelspec = mech4import[mech4import[:,1]==target_spec[j,1]]
            target_molwght   = molecular_wghts[mech4import[:,1]==target_spec[j,1]]
            for k in range(len(target_modelspec)):
                gspro_row = pd.Series(data={'PROFILE':prof,'INPUT.POLL':i_poll,
                                            'MODEL.SPECIES':target_modelspec[k,2],
                                            'MASS.FRACTION':(target_modelspec[k,3].astype(np.float) * target_molwght[k].astype(np.float)) / np.sum(target_modelspec[:,3].astype(np.float) * target_molwght.astype(np.float)) * spec_fraction,
                                            'MOLECULAR.WGHT':target_molwght[k]})
                dfgspro   = dfgspro.append(gspro_row,ignore_index=True)
    
    dfgspro = dfgspro.groupby(['PROFILE','INPUT.POLL','MODEL.SPECIES','MOLECULAR.WGHT'],as_index=False).sum()
    dfgspro['MOLECULAR.WGHT'] = dfgspro['MOLECULAR.WGHT'].astype(float).apply(lambda x: '%.6E' % x)
    dfgspro['MASS.FRACTION']  = dfgspro['MASS.FRACTION'].astype(float).apply(lambda x: '%.6E' % x)
    dfgspro['MASS.FRACTION1'] = dfgspro['MASS.FRACTION']
    dfgspro = dfgspro[['PROFILE','INPUT.POLL','MODEL.SPECIES','MASS.FRACTION','MOLECULAR.WGHT','MASS.FRACTION1']]

    ### Output gscnv df to file
    today = date.today()
    dfgspro.to_csv('./output/gspro.'+MECHANISM+'_criteria.CMAQ.'+str(today)+'.txt',index=False,header=False)

####################################################################################################
def format_and_header(MECHANISM):

    ################################################################################################
    ### Import gscnv csv file.
    today    = date.today()
    f1_gspro = np.genfromtxt('./output/gspro.'+MECHANISM+'_criteria.CMAQ.'+str(today)+'.txt',delimiter=',',dtype='str')
    ################################################################################################
    
    ################################################################################################
    ###
    headerline1   = '#SPTOOL_AQM          CMAQ'
    headerline2   = '#SPTOOL_CAMX_FCRS    Not Applicable'
    headerline3   = '#SPTOOL_MECH         '+MECHANISM
    headerline4   = '#SPTOOL_RUN_TYPE     CRITERIA'
    headerline5   = '#PROFILE,INPUT.POLL,MODEL.SPECIES,MASS.FRACTION,MOLECULAR.WGHT,MASS.FRACTION'
    headerline    = '\n'.join([headerline1,headerline2,headerline3,headerline4,headerline5])
    output_file   = './output/gspro.'+MECHANISM+'_criteria.CMAQ.'+str(today)+'.txt'
    np.savetxt(output_file,f1_gspro[:],fmt='%-20s %-20s %-10s %-13s %-13s %-13s',header=headerline,comments='')
    ################################################################################################

####################################################################################################