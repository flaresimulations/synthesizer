"""Useful statistical operations not included in dependencies.

Typical usage examples:

  med = weighted_median(xs, weights)
  binned_median = binned_weighted_quantile(x, y, weights, bins,
                                           quantiles=[0.5, ])

"""
import numpy as np


def weighted_median(data, weights):
    """
    Calculates the weighted median.
    
    Args:
        data (array-like)
            The data from which to calculate the median, shape (N,).
        weights (array-like)
            The weight of each data point, shape (N,).

    Returns:
        w_median (float)
            The weighted median.
    """

    # Squeeze the arrays to remove length 1 axes
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()

    # Sort the data and weights into pairs
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))

    # Define the midpoint
    midpoint = 0.5 * sum(s_weights)

    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx:idx+2])
        else:
            w_median = s_data[idx+1]
    return w_median



def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """
    Similar to weighted median but for arbitrary quantiles.
    
    Taken from From https://stackoverflow.com/a/29677616/1718096.

    Very close to numpy.percentile, but supports weights.
    
    NOTE: quantiles should be in [0, 1]!

    Args:
        values (array-like)
             The array containing data points.
        quantiles (array-like, float)
             A list of the required quantiles in [0, 1].
        sample_weight (array-like)
             Of the same length as `array`
        values_sorted (bool)
            If True, then will skip sorting of initial array.
        old_style (bool)
            If True, will correct output to be consistent with
            numpy.percentile.
    
    Returns
        result (array-like, float)
            The computed quantiles.
    """

    # Handle the case where there are no weights
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    
    # Ensure we have arrays
    values = np.array(values)
    quantiles = np.array(quantiles)
    sample_weight = np.array(sample_weight)
    
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    # If not sorted, sort the values array
    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    # Compute all weighted quantiles 
    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight

    # Are we emulating numpy.percentile style?
    if old_style:
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)

    # Evaluate at the requested quantiles 
    result = np.interp(quantiles, weighted_quantiles, values)
        
    return result


def binned_weighted_quantile(x, y, weights, bins, quantiles):
    """
    A helper function to calculate quantiles within bins. Applies
    synthesizer.stats.weighted_quantile to each bin.

    Args:
        x (array-like)
            The x data points to be binned, shape (N,).
        y (array-like)
            The y data points to calulcate the quantiles from, shape (N,).
        bins (array-like)
            The bin edges.
        quantiles (array-like, float)
            A list of the required quantiles in [0, 1].

    Returns:
        out (array-like, float)
            The quantiles in each bin. Empty bins will contain a NaN value.
    """

    # Create an array for each bin
    out = np.full((len(bins)-1, len(quantiles)), np.nan)

    # Loop over bins
    for i, (b1, b2) in enumerate(zip(bins[:-1], bins[1:])):

        # Mask out values outside this bin
        mask = (x >= b1) & (x < b2)

        # If the bin isn't empty compute the quantiles
        if np.sum(mask) > 0:
            out[i, :] = weighted_quantile(
                y[mask], quantiles, sample_weight=weights[mask]
            )

    return np.squeeze(out)


def n_weighted_moment(values, weights, n):
    """
    Calculates the n-th moment of an array of values.

    Args:
        values (array-like)
            The values from which to calculate the moment, shape (N,).
        weights (array-like)
            The weight of each data point in values, shape (N,).
        n (int)
            The n-th order of the desired moment.

    Returns
        float
            The n-th moment.
    """

    # Ensure arguments are valid
    assert n > 0 & (values.shape == weights.shape)

    # Calculate the weight average
    w_avg = np.average(values, weights=weights)

    # Calculate the variance
    w_var = np.sum(weights * (values - w_avg) ** 2) / np.sum(weights)

    # Handle trivial cases
    if n == 1:
        return w_avg
    elif n == 2:
        return w_var
    
    else:
        
        # Compute and return the n-th moment
        w_std = np.sqrt(w_var)
        return np.average(((values - w_avg) / w_std) ** n, weights=weights)
