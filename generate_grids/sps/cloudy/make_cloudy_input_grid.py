"""
Create a grid of cloudy models
"""

import numpy as np
import argparse
from pathlib import Path
import yaml

from synthesizer.abundances import Abundances
from synthesizer.grid import Grid
from synthesizer.cloudy import create_cloudy_input, ShapeCommands

from utils import apollo_submission_script, get_grid_properties

from copy import deepcopy


def load_grid_params(param_file='c17.03', dir = 'params'):
    """
    parameters from a single param_file

    Parameters
    ----------
    param_file : str
        Location of YAML file.

    Returns
    -------
    dict
        Dictionary of cloudy parameters
    """

    # open paramter file
    with open(f'{dir}/{param_file}.yaml', "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    grid_params = {}
    fixed_params = {}

    for k, v in params.items():
        if isinstance(v, list):
            grid_params[k] = v
        else:
            fixed_params[k] = v

    return fixed_params, grid_params






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run a grid of SPS cloudy models')
    parser.add_argument("-machine", type=str, required=True) # machine (for submission script generation)
    parser.add_argument("-synthesizer_data_dir", type=str, required=True) # path to synthesizer_data_dir
    parser.add_argument("-sps_grid", type=str, required=True) # grid_name, used to define parameter file
    parser.add_argument("-cloudy_grid", type=str, required=False, default='params/default.yaml') # the parameters of the cloudy run, including any grid axes
    parser.add_argument("-cloudy_path", type=str, required=True) # path to cloudy directory (not executable; this is assumed to {cloudy}/{cloudy_version}/source/cloudy.ext)
    args = parser.parse_args()


    # load the cloudy parameters you are going to run
    fixed_params, grid_params = load_grid_params(args.cloudy_grid)


    # open the parent sps grid
    sps_grid = Grid(args.sps_grid, grid_dir=f'{args.synthesizer_data_dir}/grids')

    # define output directories 
    output_dir = f'{synthesizer_data_dir}/sps/cloudy/{args.sps_grid}_cloudy-{args.cloudy_grid}'

    # make output directories 
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # for submission system output files
    Path(f'{output_dir}/output').mkdir(parents=True, exist_ok=True)

    # add the SPS grid parameters to grid_params
    for axis in sps_grid.axes:
        grid_params[axis] = sps_grid.bin_centres[axis]

    # print fixed parameters
    for k, v in fixed_params.items():
        print(k,v)

    # print grid parameters, including SPS parameters
    for k, v in grid_params.items():
        print(k,v)

    # get the indices of the reference grid point (this is used by the reference model)
    sps_ref_grid_point = sps_grid.get_grid_point([cloudy_params[k+'_ref'] for k in sps_grid.axes])

    # add these to the parameter file
    for k, i in zip(sps_grid.axes, sps_ref_grid_point):
        cloudy_params[k+'_ref'] = i


    # save all parameters
    yaml.dump(cloudy_params, open(f'{output_dir}/params.yaml', 'w'))

    # get properties of the grid
    axes, n_axes, shape, n_models, mesh, model_list, index_list = get_grid_properties(grid_params, verbose = True)

    # loop over all models
    for i, (grid_params_, grid_index_) in enumerate(zip(model_list, index_list)):
    
        # get a dictionary of all parameters
        grid_params = dict(zip(axes, grid_params_))

        # get a dictionary of the parameter grid point
        grid_index = dict(zip(axes, grid_index_))

        # get a dictionary of just the SPS parameters
        sps_params = {k:grid_params[k] for k in sps_grid.axes}

        # get a dictionary of the SPS parameter grid point
        sps_index = {k:grid_index[k] for k in sps_grid.axes}

        # get a tuple of the SPS grid point
        sps_grid_point = tuple(grid_index[k] for k in sps_grid.axes)

        # join the fixed and grid parameters 
        cloudy_params = fixed_params | grid_params

        # set cloudy metallicity parameter to the stellar metallicity
        if 'metallicities' in sps_grid.axes:
            cloudy_params['Z'] = sps_params['metallicities']
        elif 'log10metallicities' in sps_grid.axes:
            cloudy_params['Z'] = 10**sps_params['log10metallicities']
        
        # create abundances object
        abundances = Abundances(cloudy_params['Z'], d2m=cloudy_params['d2m'], alpha=cloudy_params['alpha'], N=cloudy_params['N'], C=cloudy_params['C'])

        # if reference U model is used
        if cloudy_params['U_model'] == 'ref':

            # calculate the difference between the reference log10Q (LyC continuum luminosity) and the current grid point
            delta_log10Q = sps_grid.log10Q['HI'][sps_grid_point] - sps_grid.log10Q['HI'][sps_ref_grid_point]

            # for spherical geometry the effective log10U is this
            if cloudy_params['geometry'] == 'spherical':

                log10U = cloudy_params['log10U_ref'] + (1/3) * delta_log10Q

            # for plane-parallel geometry the effective just scales with log10Q
            elif cloudy_params['geometry'] == 'planeparallel':

                log10U = cloudy_params['log10U_ref'] + delta_log10Q

            else:

                print(f"ERROR: do not understand geometry choice: {cloudy_params['geometry']}")

        # if fixed U model is used
        elif cloudy_params['U_model'] == 'fixed':

            log10U = cloudy_params['log10U_ref']

        else:

            print(f"ERROR: do not understand U model choice: {cloudy_params['U_model']}")

        # set log10U to provide cloudy
        cloudy_params['log10U'] = float(log10U)

        # get wavelength
        lam = sps_grid.lam # AA

        # get luminosity 
        lnu = grid.spectra['stellar'][sps_grid_point]

        # this returns the relevant shape commands, in this case for a tabulated SED
        shape_commands = ShapeCommands.table_sed(str(i), lam, lnu,  output_dir=output_dir)

        # create cloudy input file
        create_cloudy_input(str(i), shape_commands, abundances, output_dir = output_dir, **cloudy_params)

    # create submission script
    if args.machine == 'apollo':
        apollo_submission_script(n_models, output_dir, args.cloudy_path, cloudy_params['cloudy_version'])
