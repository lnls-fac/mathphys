"""Useful algorithms."""

import numpy as _np


def create_meshgrid_zigzag(*arrays):
    """Create N-D Zig-Zag mesh grid from arrays.

    The code ensures that the trajectory followed is continuous, in the
    sense that if we want to map all the points of an hyper-cube in N-D
    space:
        >>> x = [1, 2, 3]
        >>> N = 5
        >>> arrays = [x, ] * N
        >>> points = np.array(create_meshgrid_zigzag(*arrays))
        >>> print(points.shape == (N, len(x) ** N)
        True
        >>> np.linalg.norm(np.diff(points, axis=1), axis=0) == 1
        True

    In the simple case of a 2-D grid, we have:
        >>> x = [1, 2, 3]
        >>> y = [1, 2, 3, 4]
        >>> X, Y = create_meshgrid_zigzag(x, y)
        >>> points = np.array([X, Y])
        >>> points
        array([[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            [1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3, 4]])

    Note that X varies slowly, while Y is fast.

    Args:
        *arrays: sequence of 1D arrays to create meshgrid.

    Returns:
        grids: list of numpy.ndarray with grid coordinates. Each array is
            1D, with size equal to the product of the sizes of the input
            arrays.

    """
    # NOTE: This code was done by Google AI in 2026/10/15. It takes some
    # time to digest, but if you go through the details, it makes sense.
    # Besides, I tested it using the example of the docstring and others
    # and it gives consistent results when checked for the distance
    # between successive points shown in the docstring. If you want to
    # learn more about this topic, search for "Boustrophedon trajectory".
    ndims = len(arrays)
    if ndims == 0:
        return ()

    arrays = [_np.asarray(arr) for arr in arrays]
    shapes = [arr.size for arr in arrays]

    # 1. Start from the last dimension of the list (last array)
    curr_indices = _np.arange(shapes[-1])[:, None]

    # 2. Build tree indices from bottom-up (from last to first dimension)
    for i in reversed(range(ndims - 1)):
        shape_i = shapes[i]
        prev_size, cum_dims = curr_indices.shape

        # Get exact memory block for repetitions
        cum_indices = _np.empty((shape_i, prev_size, cum_dims), dtype=int)

        # Vectorized trick: even slices keep order, odd ones are inverted
        cum_indices[0::2] = curr_indices
        # Invert whole block from first axis (prev_size)
        cum_indices[1::2] = curr_indices[::-1]

        # Combine slices from 3D block into 2D array
        cum_indices = cum_indices.reshape(-1, cum_dims)

        # Create and repeat indices corresponding to current dimension
        curr_dim_indices = _np.repeat(_np.arange(shape_i), prev_size)

        # Stack current dimension to the left of sub-dims. already created.
        curr_indices = _np.hstack([curr_dim_indices[:, None], cum_indices])

    # 3. Map matrix of indices back to the real values of the input arrays
    return tuple(arrays[k][curr_indices[:, k]] for k in range(ndims))
