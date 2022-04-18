import sys
from datetime import datetime
import numpy as np

startTime = datetime.now()

####################################################################################################
### S2S Tool: U.S. EPA's SPECIATE-to-SMOKE Tool translates data from the U.S. EPA's SPECIATE
### database into files needed by SMOKE to generate gridded emission files for photochemical
### modeling. SPECIATE is a database of organic gas, particle, and mercury speciation profiles.
### These profiles provide greater specificity than what is needed for a chemical mechanism
### within a photochemical model, such as CMAQ. The S2S Tool bridges this gap and translates
### SPECIATE data into a format that is chemical mechanism specific.
####################################################################################################

####################################################################################################
### User Input
### Select photochemical modeling mechanism.
### Currently accepts CB6R3_AE7, CB6R3_AE8, CB6R5_AE7, CB6R5_AE8, CRACMMv0.3
MECHANISM  = 'CB6R3_AE8'
### Select version of SPECIATE
SPECIATE   = 'SPECIATEv5.2'
### Location of modules:
sys.path.append('./modules/')
####################################################################################################

####################################################################################################
### Import all input data.
profiles         = np.genfromtxt("./input/export_profiles.csv",delimiter=",",skip_header=1,dtype='str')
species          = np.genfromtxt("./input/export_species.csv",delimiter=",",skip_header=1,dtype='str')
species_props    = np.genfromtxt("./input/export_species_properties.csv",delimiter=",",skip_header=1,dtype='str')
molecular_wghts  = np.genfromtxt("./input/mechanism_molwghts.csv",delimiter=",",skip_header=1,dtype='str')
mech4import      = np.genfromtxt("./input/mechanism_forImport_"+SPECIATE+".csv",delimiter=",",skip_header=1,dtype='str')
#chem_props_vars  = np.genfromtxt("./input/chemical_assignments.csv",delimiter=",",skip_header=1,usecols=(1,4,5,6,9,12,13,14))  # SPECIATE_ID, NumC, NumO, MW, Koa, log(C*), SOA Yield, MIR
#chem_props_strs  = np.genfromtxt("./input/chemical_assignments.csv",delimiter=",",skip_header=1,dtype='str',usecols=(0,2,3))   # GROUP, HAPS, nonVOCTOG
####################################################################################################

####################################################################################################
### Filter Import data for the target MECHANISM.
molecular_wghts  = molecular_wghts[molecular_wghts[:,0]==MECHANISM]
mech4import      = mech4import[mech4import[:,0]==MECHANISM]
####################################################################################################

####################################################################################################
### This module contains several functions that perform QA checks on the input files.
import check_inputs
### This module contain a single function that checks for necessary output directories and creates them, if absent.
import check_directories
### This module generates the gscnv file for the target MECHANISM.
import gscnv
### This module generates the gspro file for the target MECHANISM.
import gspro
####################################################################################################

### Create gscnv file for the target MECHANISM
gscnv.gen_gscnv(profiles,MECHANISM)
### Formats and adds header to gscnv file
gscnv.format_and_header(MECHANISM)

### Creates an array of molecular weights for entries in mech4import array
molecular_wghts = gspro.append_molwght(molecular_wghts,mech4import)
### Create gspro file for the target MECHANISM
gspro.gen_gspro(profiles,species,species_props,molecular_wghts,mech4import,MECHANISM)
### Formats and adds header to gspro file
gspro.format_and_header(MECHANISM)

print("Time to run the S2S Tool: ",datetime.now() - startTime)

### QA check on subpuc_usage.csv file
#check_inputs.check_usage(subpuc_names,year,subpuc_usage)
### QA check on subpuc_usetimescales.csv file
#check_inputs.check_usetime(subpuc_names,subpuc_usetime)
### QA check on subpuc_controls.csv file
#check_inputs.check_controls(subpuc_names,subpuc_controls)
### QA check on subpuc_1st_order_speciation.csv file
#check_inputs.check_1st_order_spec(subpuc_names,first_ord_spec)
### QA check on subpuc_organic_speciation.csv file
#check_inputs.check_organic_spec(subpuc_names,organic_spec,chem_index)
### QA check on chemical_assignments.csv file
#check_inputs.check_chem_assignments(chem_props_vars,chem_props_strs,chem_index)
### QA check on subpuc_SCC_map.csv file
#check_inputs.check_subpuc_SCC_map(subpuc_scc_map,subpuc_names)

### Checks for necessary output directories and creates them, if absent.
#check_directories.check_create_directory(year)