/******************************************************************************
 * C extension to calculate SEDs for star particles.
 * Calculates weights on an arbitrary dimensional grid given the mass.
 *****************************************************************************/
/* C includes */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Python includes */
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

/* Local includes */
#include "macros.h"
#include "threadpool.h"
#include "weights.h"

// Declare a mutex for the spectra array
std::mutex spectraMutex;

/**
 * @brief This calculates the spectra of a particle using a cloud in cell
 *        approach.
 *
 * @param grid_props: An array of the properties along each grid axis.
 * @param part_props: An array of the particle properties, in the same property
 *                    order as grid props.
 * @param mass: The mass of the current particle.
 * @param grid_spectra: The grid of SPS spectra.
 * @param dims: The length of each grid dimension.
 * @param ndim: The number of grid dimensions.
 * @param spectra: The array of particle spectra.
 * @param nlam: The number of wavelength elements.
 * @param fesc: The escape fraction.
 * @param p: The index of the current particle.
 */
void spectra_loop_cic(const double **grid_props, const double **part_props,
                      const double mass, const double *grid_spectra,
                      const int *dims, const int ndim, double *spectra,
                      const int nlam, const double fesc, const int p) {

  /* Setup the index and mass fraction arrays. */
  int part_indices[ndim];
  double axis_fracs[ndim];

  /* Loop over dimensions finding the mass weightings and indicies. */
  for (int dim = 0; dim < ndim; dim++) {

    /* Get this array of grid properties for this dimension */
    const double *grid_prop = grid_props[dim];

    /* Get this particle property. */
    const double part_val = part_props[dim][p];

    /* Here we need to handle if we are outside the range of values. If so
     * there's no point in searching and we return the edge nearest to the
     * value. */
    int part_cell;
    double frac;
    if (part_val <= grid_prop[0]) {

      /* Use the grid edge. */
      part_cell = 0;
      frac = 1;

    } else if (part_val > grid_prop[dims[dim] - 1]) {

      /* Use the grid edge. */
      part_cell = dims[dim] - 1;
      frac = 1;

    } else {

      /* Find the grid index corresponding to this particle property. */
      part_cell =
          binary_search(/*low*/ 0, /*high*/ dims[dim] - 1, grid_prop, part_val);

      /* Calculate the fraction. Note, here we do the "low" cell, the cell
       * above is calculated from this fraction. */
      frac = (grid_prop[part_cell] - part_val) /
             (grid_prop[part_cell] - grid_prop[part_cell - 1]);
    }

    /* Set the fraction for this dimension. */
    axis_fracs[dim] = (1 - frac);

    /* Set this index. */
    part_indices[dim] = part_cell;
  }

  /* To combine fractions we will need an array of dimensions for the subset.
   * These are always two in size, one for the low and one for high grid
   * point. */
  int sub_dims[ndim];
  for (int idim = 0; idim < ndim; idim++) {
    sub_dims[idim] = 2;
  }

  /* Now loop over this collection of cells collecting and setting their
   * weights. */
  for (int icell = 0; icell < (int)pow(2, (double)ndim); icell++) {

    /* Set up some index arrays we'll need. */
    int subset_ind[ndim];
    int frac_ind[ndim];

    /* Get the multi-dimensional version of icell. */
    get_indices_from_flat(icell, ndim, sub_dims, subset_ind);

    /* Multiply all contributing fractions and get the fractions index
     * in the grid. */
    double frac = 1;
    for (int idim = 0; idim < ndim; idim++) {
      if (subset_ind[idim] == 0) {
        frac *= (1 - axis_fracs[idim]);
        frac_ind[idim] = part_indices[idim] - 1;
      } else {
        frac *= axis_fracs[idim];
        frac_ind[idim] = part_indices[idim];
      }
    }

    /* Early skip for cells contributing a 0 fraction. */
    if (frac <= 0)
      continue;

    /* We have a contribution, get the flattened index into the grid array. */
    const int grid_ind = get_flat_index(frac_ind, dims, ndim);

    /* Define the weight. */
    double weight = mass * frac;

    /* Get the spectra ind. */
    int unraveled_ind[ndim + 1];
    get_indices_from_flat(grid_ind, ndim, dims, unraveled_ind);
    unraveled_ind[ndim] = 0;
    int spectra_ind = get_flat_index(unraveled_ind, dims, ndim + 1);

    {
      std::unique_lock<std::mutex> lock(spectraMutex);
      /* Add this grid cell's contribution to the spectra */
      for (int ilam = 0; ilam < nlam; ilam++) {

        /* Add the contribution to this wavelength. */
        spectra[p * nlam + ilam] +=
            grid_spectra[spectra_ind + ilam] * (1 - fesc) * weight;
      }
    }
  }
}

/**
 * @brief This calculates the spectra of a particle using a nearest grid point
 *        approach.
 *
 * @param grid_props: An array of the properties along each grid axis.
 * @param part_props: An array of the particle properties, in the same property
 *                    order as grid props.
 * @param mass: The mass of the current particle.
 * @param grid_spectra: The grid of SPS spectra.
 * @param dims: The length of each grid dimension.
 * @param ndim: The number of grid dimensions.
 * @param spectra: The array of particle spectra.
 * @param nlam: The number of wavelength elements.
 * @param fesc: The escape fraction.
 * @param p: The index of the current particle.
 */
void spectra_loop_ngp(const double **grid_props, const double **part_props,
                      const double mass, const double *grid_spectra,
                      const int *dims, const int ndim, double *spectra,
                      const int nlam, const double fesc, const int p) {

  /* Setup the index array. */
  int part_indices[ndim];

  /* Loop over dimensions finding the indicies. */
  for (int dim = 0; dim < ndim; dim++) {

    /* Get this array of grid properties for this dimension */
    const double *grid_prop = grid_props[dim];

    /* Get this particle property. */
    const double part_val = part_props[dim][p];

    /* Here we need to handle if we are outside the range of values. If so
     * there's no point in searching and we return the edge nearest to the
     * value. */
    int part_cell;
    if (part_val <= grid_prop[0]) {

      /* Use the grid edge. */
      part_cell = 0;

    } else if (part_val > grid_prop[dims[dim] - 1]) {

      /* Use the grid edge. */
      part_cell = dims[dim] - 1;

    } else {

      /* Find the grid index corresponding to this particle property. */
      part_cell =
          binary_search(/*low*/ 1, /*high*/ dims[dim] - 1, grid_prop, part_val);
    }

    /* Set the index. */
    part_indices[dim] = part_cell;
  }

  /* Get the weight's index. */
  const int grid_ind = get_flat_index(part_indices, dims, ndim);

  /* Get the spectra ind. */
  int unraveled_ind[ndim + 1];
  get_indices_from_flat(grid_ind, ndim, dims, unraveled_ind);
  unraveled_ind[ndim] = 0;
  int spectra_ind = get_flat_index(unraveled_ind, dims, ndim + 1);

  /* Add this grid cell's contribution to the spectra */
  for (int ilam = 0; ilam < nlam; ilam++) {

    /* Add the contribution to this wavelength. */
    spectra[p * nlam + ilam] +=
        grid_spectra[spectra_ind + ilam] * (1 - fesc) * mass;
  }
}

typedef struct {
  const double *part_mass;
  const double **grid_props;
  const double **part_props;
  const double *grid_spectra;
  const int *dims;
  int ndim;
  double *spectra;
  int nlam;
  double fesc;
  const char *method;
} MapperData;

void spectra_mapper(void *map_data, int n_elements, void *extra_data) {

  /* Unpack the extra data. */
  MapperData *mapper_data = (MapperData *)extra_data;
  const double *part_mass = mapper_data->part_mass;
  const double **grid_props = mapper_data->grid_props;
  const double **part_props = mapper_data->part_props;
  const double *grid_spectra = mapper_data->grid_spectra;
  const int *dims = mapper_data->dims;
  const int ndim = mapper_data->ndim;
  double *spectra = mapper_data->spectra;
  const int nlam = mapper_data->nlam;
  const double fesc = mapper_data->fesc;
  const char *method = mapper_data->method;
  const int *inds = reinterpret_cast<const int *>(map_data);

  for (int i = 0; i < n_elements; i++) {

    /* Get the index into the particle data. */
    const int p = inds[i];

    /* Get this particle's mass. */
    const double mass = part_mass[p];

    /* Finally, compute the spectra for this particle using the
     * requested method. */
    if (strcmp(method, "cic") == 0) {
      spectra_loop_cic(grid_props, part_props, mass, grid_spectra, dims, ndim,
                       spectra, nlam, fesc, p);
    } else if (strcmp(method, "ngp") == 0) {
      spectra_loop_ngp(grid_props, part_props, mass, grid_spectra, dims, ndim,
                       spectra, nlam, fesc, p);
    } else {
      /* Only print this warning once */
      if (p == 0)
        printf(
            "Unrecognised gird assignment method (%s)! Falling back on CIC\n",
            method);
      spectra_loop_cic(grid_props, part_props, mass, grid_spectra, dims, ndim,
                       spectra, nlam, fesc, p);
    }
  }
}

/**
 * @brief Computes an integrated SED for a collection of particles.
 *
 * @param np_grid_spectra: The SPS spectra array.
 * @param grid_tuple: The tuple containing arrays of grid axis properties.
 * @param part_tuple: The tuple of particle property arrays (in the same order
 *                    as grid_tuple).
 * @param np_part_mass: The particle mass array.
 * @param fesc: The escape fraction.
 * @param np_ndims: The size of each grid axis.
 * @param ndim: The number of grid axes.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements.
 */
PyObject *compute_particle_seds(PyObject *self, PyObject *args) {

  int ndim;
  int npart, nlam;
  int nthreads;
  double fesc;
  PyObject *grid_tuple, *part_tuple;
  PyArrayObject *np_grid_spectra;
  PyArrayObject *np_part_mass, *np_ndims;
  char *method;

  if (!PyArg_ParseTuple(args, "OOOOdOiiisi", &np_grid_spectra, &grid_tuple,
                        &part_tuple, &np_part_mass, &fesc, &np_ndims, &ndim,
                        &npart, &nlam, &method, &nthreads))
    return NULL;

  /* Quick check to make sure our inputs are valid. */
  if (ndim == 0)
    return NULL;
  if (npart == 0)
    return NULL;
  if (nlam == 0)
    return NULL;

  /* Extract a pointer to the spectra grids */
  const double *grid_spectra =
      reinterpret_cast<const double *>(PyArray_DATA(np_grid_spectra));

  /* Set up arrays to hold the SEDs themselves. */
  double *spectra =
      reinterpret_cast<double *>(malloc(npart * nlam * sizeof(double)));
  bzero(spectra, npart * nlam * sizeof(double));

  /* Extract a pointer to the grid dims */
  const int *dims = reinterpret_cast<const int *>(PyArray_DATA(np_ndims));

  /* Extract a pointer to the particle masses. */
  const double *part_mass =
      reinterpret_cast<const double *>(PyArray_DATA(np_part_mass));

  /* Allocate a single array for grid properties*/
  int nprops = 0;
  for (int dim = 0; dim < ndim; dim++)
    nprops += dims[dim];
  const double **grid_props =
      reinterpret_cast<const double **>(malloc(nprops * sizeof(double *)));

  /* How many grid elements are there? (excluding wavelength axis)*/
  int grid_size = 1;
  for (int dim = 0; dim < ndim; dim++)
    grid_size *= dims[dim];

  /* Allocate an array to hold the grid weights. */
  double *grid_weights =
      reinterpret_cast<double *>(malloc(grid_size * sizeof(double)));
  bzero(grid_weights, grid_size * sizeof(double));

  /* Set up the threadpool. */
  ThreadPool threadpool(nthreads);

  /* Unpack the grid property arrays into a single contiguous array. */
  for (int idim = 0; idim < ndim; idim++) {

    /* Extract the data from the numpy array. */
    PyArrayObject *np_grid_arr =
        reinterpret_cast<PyArrayObject *>(PyTuple_GetItem(grid_tuple, idim));
    double *grid_arr = reinterpret_cast<double *>(PyArray_DATA(np_grid_arr));

    /* Assign this data to the property array. */
    grid_props[idim] = grid_arr;
  }

  /* Allocate a single array for particle properties. */
  const double **part_props = reinterpret_cast<const double **>(
      malloc(npart * ndim * sizeof(double *)));

  /* Unpack the particle property arrays into a single contiguous array. */
  for (int idim = 0; idim < ndim; idim++) {
    /* Extract the data from the numpy array. */
    PyArrayObject *np_part_arr =
        reinterpret_cast<PyArrayObject *>(PyTuple_GetItem(part_tuple, idim));
    const double *part_arr =
        reinterpret_cast<const double *>(PyArray_DATA(np_part_arr));

    /* Assign this data to the property array. */
    part_props[idim] = part_arr;
  }

  /* Set up the extra data we need to pass to the threadpool. */
  MapperData *mapper_data =
      reinterpret_cast<MapperData *>(malloc(sizeof(MapperData)));
  mapper_data->part_mass = part_mass;
  mapper_data->grid_props = grid_props;
  mapper_data->part_props = part_props;
  mapper_data->grid_spectra = grid_spectra;
  mapper_data->dims = dims;
  mapper_data->ndim = ndim;
  mapper_data->spectra = spectra;
  mapper_data->nlam = nlam;
  mapper_data->fesc = fesc;
  mapper_data->method = method;

  // Create an array of atomic indices.
  std::atomic<int> *inds = new std::atomic<int>[npart];
  printf("npart: %d\n", npart);

  for (int i = 0; i < npart; i++) {
    inds[i].store(i);
  }

  /* Let the threadpool do its magic! */
  /* NOTE: if we don't have multiple threads the threadpool will simply loop. */
  threadpool.map(spectra_mapper, inds, npart, (int)(npart / nthreads),
                 mapper_data);
  free(mapper_data);

  /* Clean up memory! */
  free(grid_weights);
  free(part_props);
  free(grid_props);

  /* Reconstruct the python array to return. */
  npy_intp np_dims[2] = {
      npart,
      nlam,
  };
  PyArrayObject *out_spectra = (PyArrayObject *)PyArray_SimpleNewFromData(
      2, np_dims, NPY_FLOAT64, spectra);

  return Py_BuildValue("N", out_spectra);
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef SedMethods[] = {
    {"compute_particle_seds", compute_particle_seds, METH_VARARGS,
     "Method for calculating particle intrinsic spectra."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "make_particle_sed",                   /* m_name */
    "A module to calculate particle seds", /* m_doc */
    -1,                                    /* m_size */
    SedMethods,                            /* m_methods */
    NULL,                                  /* m_reload */
    NULL,                                  /* m_traverse */
    NULL,                                  /* m_clear */
    NULL,                                  /* m_free */
};

PyMODINIT_FUNC PyInit_particle_spectra(void) {
  PyObject *m = PyModule_Create(&moduledef);
  import_array();
  return m;
}
