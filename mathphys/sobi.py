"""ICA via SOBI."""

from itertools import combinations as _combinations

import numpy as _np


class SOBI():
    """ICA via Second Order Blind Source Separation.

    Implementation based on that of @edouarpineu, available at
    https://github.com/edouardpineau/Time-Series-ICA-with-SOBI-Jacobi
    + modifications to comform the code to be similar to the API of
    Scikit-learn's FastICA.

    References:
    [1] Cardoso, Souloumiac. Jacobi Angles For Simultaneous
        Diagonalization. SIAM Journal on Matrix Analysis and
        Applications. https://doi.org/10.1137/S0895479893259546
        Available at
        https://www.researchgate.net/publication/277295728_Jacobi_Angles_For_Simultaneous_Diagonalization
    [2] A. Belouchrani, K. Abed-Meraim, J. -. Cardoso and E. Moulines, "A
        blind source separation technique using second-order statistics,"
        in IEEE Transactions on Signal Processing, vol. 45, no. 2, pp.
        434-444, Feb. 1997, doi: 10.1109/78.554307.
        Available at http://pzs.dstu.dp.ua/DataMining/ica/bibl/Belouchrani.pdf
    """

    def __init__(
        self,
        n_components=None,
        n_lags=5,
        tol=1e-5,
        max_iter=1000,
        whiten="unit-variance",
        isreal=True,
        verbose=False
    ):
        """."""
        self.n_components = n_components
        self.tol = tol
        self.n_lags = n_lags
        self.max_iter = max_iter
        self.whiten = whiten
        self.isreal = isreal
        self.verbose = verbose
        self._isfitted = False

        self.covs = None
        self.covs_diag = None
        self.mean_ = None
        self.mixing_ = None
        self.whitening_ = None
        self.components_ = None

    def _fit_transform(self, X, compute_sources=False):
        """."""
        # input data w/ column-vectors convention
        n_samples, n_features = X.shape
        n_components = self.n_components
        rank = min(n_samples, n_features)
        if n_components is None or n_components > rank:
            n_components = rank
            print("Warning: n_components is None or larger than data rank")
            print("Using n_components = min(X.shape)")

        # calculations w/ row-vectors convention

        XT, K, X_mean = self.whiten_data(X.T)
        covs = self.get_covariances(XT)
        covs = covs + 1j * _np.zeros_like(covs)
        # cast to complex to avoid errors involving
        # multipications of possibly complex matrices.
        W, covs_diag = self.joint_diag(
                            covs,
                            W_init=None,
                            tol=self.tol,
                            max_iter=self.max_iter,
                            verbose=self.verbose
                        )

        self.covs = covs
        self.covs_diag = covs_diag
        if self.isreal:
            self.covs_diag = _np.real(self.covs_diag)
            W = _np.real(W)

        if compute_sources:
            S = _np.dot(W.T, XT).T
        else:
            S = None

        if self.whiten == "unit-variance":
            S_std = S.std(axis=0, keepdims=True)
            S /= S_std
            W /= S_std.T

        self.components_ = _np.dot(W.T, K)
        self.mean_ = X_mean
        self.whitening_ = K

        self.mixing_ = _np.linalg.pinv(self.components_)
        self._unmixing = W.T

        return S

    def fit_transform(self, X):
        """."""
        return self._fit_transform(X, compute_sources=True)

    def fit(self, X):
        """."""
        self._fit_transform(X, compute_sources=False)
        return

    def transform(self, X):
        """."""
        return _np.dot(X, self.components_.T)

    def inverse_transform(self, X):
        """."""
        if self.mixing_ is not None:
            X = _np.dot(X, self.mixing_.T)
            X += self.mean_[None, :]
            return X
        else:
            print("Data must first be fitted.")

    def get_covariances(self, X, n_lags=None):
        """Calculate the time-lagged self-covariances of data matrix X.

        Centers the data about its mean prior to calulation.

        Args:
            X (n,m-array): data matrix, data samples as row-vectors
            n_lags (int, optional): Number of time-lags.

        Returns:
            (n_lags + 1, m, m)-array: containing the n_lags + 1 m x m
            time-lagged self-covariances
        """
        m, n = X.shape
        lags = n_lags or self.n_lags
        n_ = n - lags

        timelagged_covs = _np.empty((lags+1, m, m))
        X_0, _ = self.center(X[:, :n_])

        for k in range(lags + 1):
            X_k, _ = self.center(X[:, k:k + n_])
            cov = _np.dot(X_0, X_k.T)
            cov /= n_
            timelagged_covs[k] = cov

        return timelagged_covs

    def whiten_data(self, X, n_components=None):
        """Compress & whiten data.

        Projects into the n_components-dimensional space
        of principal components z = S^-1 U^T X. Assumes data vectors as rows.

        Args:
            X ((m,n)-array): data matrix, data vecotrs (samples) as rows.
            n_components (int, optional): number of principal components.
            Defaults to None, in which case the value set at self.n_components
            is consulted.

        Returns:
            X_white ((n_components, m)-array): whitened & compressed data
            K ((n_components, m)-array): whitening matrix.
            X_mean (m-array): mean value for each sample of X.
        """
        n_components = n_components or self.n_components
        Xnm, X_mean = self.center(X)
        U, s, _ = _np.linalg.svd(Xnm, full_matrices=False)
        K = (U / s).T[:n_components]
        X_white = K @ X  # z = U.T S^-1 X

        return X_white, K, X_mean

    def center(self, X):
        """Center data about its mean.

        Args:
            X ((m,n)-array): input data, data vectors (samples) as rows.

        Returns:
            Xnm ((m,n)-array): mean-subtracted data matrix.
            X_mean (m-array): mean value for each sample.
        """
        X_ = X.copy()
        X_mean = X_.mean(axis=-1)
        Xnm = X_ - X_mean[:, None]
        return Xnm, X_mean

    def off(self, M):
        """Compute the sum of squared off-diag entries of a matrix.

        Args:
            M (m,n-array): input matrix.

        Returns:
            off(M) (float): the sum of squares of the off-diagonal entries of
            M.
        """
        mat = M.copy()
        mat -= _np.diag(_np.diag(mat))
        return _np.linalg.norm(mat, ord='fro')**2

    def jacobi_rotation(self, M):
        """Perform Jacobi rotation [1] for joint diagonalization.

        Closed-form angles of Jacobi rotation for joint diagonalization is
        found in reference [2].

        Args:
            M ((l, 2 ,2)-array): containing the (i, j) Jacobi block of the
            l matrices to be jointly diagonalized.

        Returns:
            R_ijcs ((2,2)-array): Jacobi rotation matrix for the the l (i,j)
            Jacobi block.

        Refs.:
            [1] https://en.wikipedia.org/wiki/Jacobi_rotation
            [2] Cardoso, Souloumiac. Jacobi Angles For Simultaneous
                Diagonalization. SIAM Journal on Matrix Analysis and
                Applications. https://doi.org/10.1137/S0895479893259546
                Available at
                https://www.researchgate.net/publication/277295728_Jacobi_Angles_For_Simultaneous_Diagonalization
        """
        h = _np.array([M[:, 0, 0] - M[:, 1, 1],
                       M[:, 0, 1] + M[:, 1, 0],
                       1j*(M[:, 1, 0] - M[:, 0, 1])]).T  # [2], eq. (5)

        G = _np.real(_np.dot(h.T, h))  # [2], eq. (4)
        _, eigenvecs = _np.linalg.eigh(G)
        x, y, z = eigenvecs[:, -1]
        if x < 0:
            x, y, z = -x, -y, -z  # inner-rotations remark on [2]

        r = _np.linalg.norm([x, y, z])  # [2], eqs (6)
        c = _np.sqrt((x + r) / 2 / r)
        s = (y - 1j * z) / _np.sqrt(2 * r * (x + r))

        R_ijcs = _np.array([[c, _np.conjugate(s)],
                           [-s, _np.conjugate(c)]])  # [2], eq (2)

        return R_ijcs

    def joint_diag(
        self, matrices, tol, max_iter, W_init=None, verbose=False
    ):
        """Joint-diagonalize a set of symmetric, square matrices.

        Does so by iteratively applying Jacobi rotations, as described in
        Cardoso & Souloumiac. Jacobi Angles For Simultaneous Diagonalization.
        SIAM Journal on Matrix Analysis and Applications.
        https://www.researchgate.net/publication/277295728_Jacobi_Angles_For_Simultaneous_Diagonalization

        Args:
            matrices ((l, m, m)-array): the l m x m matrices to be
            jointly-diagonalized.

            W_init ((m, m)-array): the prior guess for the rotation matrix
            diagonalizing the set of matrices.

            tol (float): tolerance precision. If the objective function
            (sum of squared off-diagonal entries) does not decrease by more
            than tol at a given iteration, the routine finishes.

            max_iter (int): maximum number of iterations.

            verbose (bool, optional): whether to log the iterations counter
            and objective function value at each iteration. Defaults to False.

        Returns:
            W ((m,m)-array): rotation matrix that joint diagonalizes the
            set of matrices.

            matrices ((l, m, m)-array): the l mxm diagonalized matrices.
        """
        nr_rows = matrices.shape[1]
        ij_pairs = list(_combinations(range(nr_rows), 2))

        identity = _np.eye(nr_rows) + 1j * _np.zeros((nr_rows, nr_rows))
        W = identity if W_init is None else W_init

        objective_func = _np.sum([self.off(mat) for mat in matrices])

        count = 0

        if verbose:
            print("Joint-diagonalizing...")
            print(f"iter {count:3d} \t objective: {objective_func:.2g}")

        diff = _np.inf

        while diff > tol and count < max_iter:
            count += 1

            for (i, j) in ij_pairs:
                W_ = identity.copy()
                submat_idcs = _np.ix_([i, j], [i, j])
                idx = (slice(None), ) + submat_idcs
                R = self.jacobi_rotation(matrices[idx])
                W_[submat_idcs] = _np.dot(W_[submat_idcs], R)
                W = _np.dot(W, W_.T)
                # V = V_.copy()
                # matrices = _np.matmul(_np.matmul(V, matrices), V.T)
                matrices = W_ @ (matrices @ W_.T)

            objective_func_ = _np.sum([self.off(mat) for mat in matrices])

            diff = _np.abs(objective_func - objective_func_)

            if verbose:
                print(f"iter {count:3d} \t objective: {objective_func:.2f}")

            objective_func = objective_func_

        return W, matrices
