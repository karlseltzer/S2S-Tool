import os
import numpy as np

####################################################################################################
### This function checks for necessary output directories and creates them, if absent.
####################################################################################################

####################################################################################################
def check_create_directory(year):
    try:
        os.makedirs('./output/air_quality_potential/'+str(year))
    except FileExistsError:
        pass
    try:
        os.makedirs('./output/emissions_by_subpuc/'+str(year))
    except FileExistsError:
        pass
    try:
        os.makedirs('./output/emissions_by_scc/'+str(year))
    except FileExistsError:
        pass
    try:
        os.makedirs('./output/emissions_spatially_allocated/'+str(year))
    except FileExistsError:
        pass
    try:
        os.makedirs('./output/speciated_emissions_spatially_allocated/'+str(year))
    except FileExistsError:
        pass
    try:
        os.makedirs('./output/smoke_flat_file/'+str(year))
    except FileExistsError:
        pass
####################################################################################################