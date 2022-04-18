import sys
import numpy as np
from datetime import date,datetime
import pandas as pd

####################################################################################################
### This function generates a gscnv file for the target MECHANISM.
####################################################################################################

####################################################################################################
def gen_gscnv(profiles,MECHANISM):
    
    ### gscnv file columns
    column_names  = ['INPUT.POLL','OUTPUT.POLL','PROFILE','OUTPUT.MASS/INPUT.MASS']
    dfgscnv       = pd.DataFrame(columns=column_names)

    for i in range(len(profiles)):
        i_poll    = 'VOC'
        o_poll    = 'TOG'
        prof      = profiles[i,0]
        ratio     = profiles[i,2]

        gscnv_row = pd.Series(data={'INPUT.POLL':i_poll,'OUTPUT.POLL':o_poll,
                                      'PROFILE':prof,'OUTPUT.MASS/INPUT.MASS':ratio})
        dfgscnv   = dfgscnv.append(gscnv_row,ignore_index=True)

    ### Output gscnv df to file
    today = date.today()
    dfgscnv.to_csv('./output/gscnv.'+MECHANISM+'_criteria.CMAQ.'+str(today)+'.txt',index=False,header=False)

####################################################################################################
def format_and_header(MECHANISM):

    ################################################################################################
    ### Import gscnv csv file.
    today    = date.today()
    f1_gscnv = np.genfromtxt('./output/gscnv.'+MECHANISM+'_criteria.CMAQ.'+str(today)+'.txt',delimiter=',',dtype='str')
    ################################################################################################
    
    ################################################################################################
    ###
    headerline1   = '#SPTOOL_AQM          CMAQ'
    headerline2   = '#SPTOOL_CAMX_FCRS    Not Applicable'
    headerline3   = '#SPTOOL_MECH         '+MECHANISM
    headerline4   = '#SPTOOL_RUN_TYPE     CRITERIA'
    headerline5   = '#INPUT.POLL,OUTPUT.POLL,PROFILE,OUTPUT.MASS/INPUT.MASS'
    headerline    = '\n'.join([headerline1,headerline2,headerline3,headerline4,headerline5])
    output_file   = './output/gscnv.'+MECHANISM+'_criteria.CMAQ.'+str(today)+'.txt'
    np.savetxt(output_file,f1_gscnv[:],fmt='%-20s',header=headerline,comments='')
    ################################################################################################

####################################################################################################