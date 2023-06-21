"""
Create a grid of cloudy models
Also create an output HDF5 for ultimately containing the grid
"""

import numpy as np
import argparse
from pathlib import Path
import yaml
import h5py
from copy import deepcopy

# synthesiser modules
from synthesizer.abundances import Abundances
from synthesizer.grid import Grid
from synthesizer.cloudy import create_cloudy_input, ShapeCommands

# local modules
from utils import apollo_submission_script, get_grid_properties




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
    sps_grid_name = args.sps_grid
    sps_grid = Grid(sps_grid_name, grid_dir=f'{args.synthesizer_data_dir}/grids')


    # get name of new grid (concatenation of sps_grid and cloudy parameter file)
    new_grid_name = f'{args.sps_grid}_cloudy-{args.cloudy_grid}'

    # define output directories 
    output_dir = f'{args.synthesizer_data_dir}/sps/cloudy/{new_grid_name}'

    # make output directories 
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # for submission system output files
    Path(f'{output_dir}/output').mkdir(parents=True, exist_ok=True)

    # set a list of the axes
    axes = list(sps_grid.axes) + list(grid_params.keys())
    print('axes:', axes)


    # add the SPS grid parameters to grid_params
    for axis in sps_grid.axes:
        grid_params[axis] = getattr(sps_grid, axis)

    # print fixed parameters
    for k, v in fixed_params.items():
        print(k,v)

    # print grid parameters, including SPS parameters
    for k, v in grid_params.items():
        print(k,v)

    # if the U model is the reference model (i.e. not fixed) save the grid point for the reference values
    if fixed_params['U_model'] == 'ref':

        # get the indices of the reference grid point (this is used by the reference model)
        sps_ref_grid_point = sps_grid.get_grid_point([fixed_params[k+'_ref'] for k in sps_grid.axes])

        # add these to the parameter file
        for k, i in zip(sps_grid.axes, sps_ref_grid_point):
            fixed_params[k+'_ref_index'] = i

    # combine all parameters
    params = fixed_params | grid_params

    # save all parameters
    yaml.dump(params, open(f'{output_dir}/params.yaml', 'w'))

    # get properties of the grid
    n_axes, shape, n_models, mesh, model_list, index_list = get_grid_properties(axes, grid_params, verbose = True)

    
    # create new synthesizer grid to contain the new grid    

    # open the new grid
    with h5py.File(f'{args.synthesizer_data_dir}/grids/{new_grid_name}.hdf5', 'w') as hf:

        # open the original SPS model grid
        with h5py.File(f'{args.synthesizer_data_dir}/grids/{sps_grid_name}.hdf5', 'r') as hf_sps:

            hf_sps.visit(print)

            # copy top-level attributes
            for k, v in hf_sps.attrs.items():
                hf.attrs[k] = v

            # add attribute with the original SPS grid axes 
            hf.attrs['sps_axes'] = hf_sps.attrs['axes']

            # copy log10Q over
            hf_sps.copy('log10Q', hf) 

        # add attribute with full grid axes
        hf.attrs['axes'] = axes

        # add the bin centres for the grid bins
        for axis in axes:
            hf[f'axes/{axis}'] = grid_params[axis]

        # add other parameters as attributes
        for k,v in params.items():
            hf.attrs[k] = v

        print('-'*50)
        print('---- attributes')
        for k,v in hf.attrs.items():
            print(k,v)
        print('---- groups and datasets')
        hf.visit(print)
        


    # loop over all models
    for i, (grid_params_tuple, grid_index_tuple) in enumerate(zip(model_list, index_list)):
    
        # get a dictionary of all parameters
        grid_params_ = dict(zip(axes, grid_params_tuple))

        # get a dictionary of the parameter grid point
        grid_index_ = dict(zip(axes, grid_index_tuple))

        # get a dictionary of just the SPS parameters
        sps_params_ = {k:grid_params_[k] for k in sps_grid.axes}

        # get a dictionary of the SPS parameter grid point
        sps_index_ = {k:grid_index_[k] for k in sps_grid.axes}

        # get a tuple of the SPS grid point
        sps_grid_point = tuple(grid_index_[k] for k in sps_grid.axes)

        # join the fixed and current iteration of the grid parameters 
        params_ = fixed_params | grid_params_

        # set cloudy metallicity parameter to the stellar metallicity
        if 'metallicity' in sps_grid.axes:
            params_['Z'] = sps_params_['metallicity']
        elif 'log10metallicity' in sps_grid.axes:
            params_['Z'] = 10**sps_params_['log10metallicity']
        
        # create abundances object
        abundances = Abundances(params_['Z'], d2m=params_['d2m'], alpha=params_['alpha'], N=params_['N'], C=params_['C'])

        # if reference U model is used
        if params_['U_model'] == 'ref':

            # calculate the difference between the reference log10Q (LyC continuum luminosity) and the current grid point
            delta_log10Q = sps_grid.log10Q['HI'][sps_grid_point] - sps_grid.log10Q['HI'][sps_ref_grid_point]

            # for spherical geometry the effective log10U is this
            if params_['geometry'] == 'spherical':

                log10U = params_['log10U_ref'] + (1/3) * delta_log10Q

            # for plane-parallel geometry the effective just scales with log10Q
            elif params_['geometry'] == 'planeparallel':

                log10U = params_['log10U_ref'] + delta_log10Q

            else:

                print(f"ERROR: do not understand geometry choice: {params_['geometry']}")

        # if fixed U model is used
        elif params_['U_model'] == 'fixed':

            log10U = params_['log10U_ref']

        else:

            print(f"ERROR: do not understand U model choice: {params_['U_model']}")

        # set log10U to provide cloudy
        params_['log10U'] = float(log10U)

        # get wavelength
        lam = sps_grid.lam # AA

        # get luminosity 
        lnu = sps_grid.spectra['stellar'][sps_grid_point]

        # this returns the relevant shape commands, in this case for a tabulated SED
        shape_commands = ShapeCommands.table_sed(str(i+1), lam, lnu,  output_dir=output_dir)

        # create cloudy input file
        create_cloudy_input(str(i+1), shape_commands, abundances, output_dir = output_dir, **params_)

    # create submission script
    if args.machine == 'apollo':
        apollo_submission_script(n_models, output_dir, args.cloudy_path, params_['cloudy_version'])
