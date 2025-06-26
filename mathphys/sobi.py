"""ICA via SOBI."""

from itertools import combinations as _combinations
import warnings

import numpy as _np


class SOBI():
    """ICA via Second Order Blind Source Separation.

    Considering a linear model for observed data

                X = S A.T,                                            (1)

    where
       - X is the n_samples x n_features data matrix (data as column vectors)
       - S is the n_samples x n_sources indepdendet source signals matrix
       - A is the n_features x n_sources linear mixture or mixing matrix
    the goal of Blind source separation, or identification, (BSS, BSI) is to
    estimate the pseudo-inverse of the mixing matrix A to determine the
    indepdent source signals S or retrive information from A.

    The Second Order Blind Identification (SOBI) algorithm estimates the
    mixing matrix by

        - projecting/whitening the data into its principal components K

                        Z = X K.T                                      (2)

        - joint-diagonalizing the n+1 time-lagged self-covariance matrices

                        C(t) = < Z(0).T Z(t) >, t = 0, ..., n          (3)

          with the linear operator W, i.e.

                        C_diag(t) = W.T C(t) W                         (4)

    Since source-signals are independent, we must have C_diag(t) =
    < S(0).T S (t)> and, from (1), (2) and (3), follows that

                C(t) = < Z(0).T Z(t) >
                     = < K X(0).T X(t) K.T >
                     = < K A S(0).T S(t) A.T K.T >
                     = K A C_diag(t) A.T K.T
    Therefore, comparing

                C_diag(t) = A' K.T C(t) K A'.T

    to (4), we find the unmixing matrix

                A' = W.T K

    This allows determination of source signals as S = X A'.T

    Construction of the matrix W joint-diagonalizing of time-lagged covariance
    matrices is achieved via successive Jacobi rotations, as described in Ref
    [1]. Ref [2] applies this diagonalization scheme directly to the blind
    source separation problem and introduces the SOBI method.

    OBS: this implementation is based on that of @edouarpineu, available at
    https://github.com/edouardpineau/Time-Series-ICA-with-SOBI-Jacobi
    Additional modifications were made to comform the code to be similar to
    the API of Scikit-learn's FastICA (convention for mixing, whitening and
    components matrices, sources normalization etc) and to make it more
    general for Blind Source Identification (selecting number of components)

    Parameters
    ----------
    n_components: int, default=None
        Number of components to use. Number of independent signals in
        the context of source separation. If None, all components are
        used, resulting in the same number of components/sources as the
        number of features of the data.

    n_lags: int, default=5
        Number of time-lags to calculate the time-lagged self-covariance
        matrices.

    tol: float, default=1e-5
        Stopping condition for the joint-diagnalization routine. If the sum
        of squares of the off-diagonal entries of the set of n_lags + 1
        self-covariance matrices decreases by less than tol, the
        diagonalization finishes.

    max_iter: int, default=1000
        Maximum number of iterations, i.e. Jacobi rotations, applied to the
        set of covariance matrices for joint-diagonalization

    whiten: str, default="unit-variance"
        Whitening convention for the source-signals.

        - If "unit-variance" the source signals S are normalized to render
            unit covariance < S.T @ S > = S.T @ S / S.shape[0] = I, where I is
            the n_components x n_components identity matrix.

        - If "arbitrary-variance", or any different string, souce-signals are
            not normalized to unity, rendering arbitrary (but diagonal)
            covariance < S.T @ S>.

    isreal: bool, default=True
        whether the input data is real-valued.

    verbose: bool, default=False
        whether to log the progress of the joint-diagonalization routine,
        showing the iterations and the corresponding value for the objective
        function (sum of squares of off-diagonal entries)

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The linear operator to apply to the data to get the independent
        sources S. For SOBI in the context of source separation, it is equal
        to the unmixing matrix, the pseudo-inverse of the mixing matrix A,
        which defines X = S A.T.

    mixing_ : ndarray of shape (n_features, n_components)
        The mixing matrix A, the pseudo-inverse of ``components_``. Linear
        operator that maps independent sources to the observed linear mixture.

    mean_ : ndarray of shape (`n_features`)
        The mean over features

    whitening_ : ndarray of shape (`n_components`, `n_components`)
        Whitening matrix for dimensionaliy reduction. It is the linear
        operator projecting data X into its principal components.


    covs : ndarray of shape (`n_lags + 1`, `n_features`, `n_features`)
        The `n_lags + 1` time-lagged covariance matrices to be
        joint-diagonalized

    covs_diag : ndarray of shape (`n_lags + 1`, `n_features`, `n_features`)
        The `n_lags + 1` joint-diagonalized time-lagged covariance matrices.



    References
    ----------
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

    Example
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as mplt
    >>> from scipy import signal
    >>> from mathphys.sobi import SOBI

    >>> # create synthetic data
    >>> np.random.seed(0)
    >>> n_samples = 2000
    >>> time = np.linspace(0, 8, n_samples)

    >>> s1 = np.sin(2 * time)                   # sine-wave
    >>> s2 = np.sign(np.sin(3 * time))          # square-wave
    >>> s3 = signal.sawtooth(2 * np.pi * time)  # saw-tooth

    >>> S = np.c_[s1, s2, s3]  # concatenate sources
    >>> S += 1e-1 * np.random.normal(size=S.shape)  # Add noise

    >>> S /= S.std(axis=0)  # Standardize data (unit covariance)

    >>> A = np.array([[1, 1, 1],
    >>>               [0.5, 2, 1.0],
    >>>               [1.5, 1.0, 2.0]])  # Mixing matrix
    >>> X = np.dot(S, A.T) # Mix data

    >>> # instantiate SOBI

    >>> sobi = SOBI(
                    n_lags=5, tol=1e-12, n_components=3,
                    whiten="unit-variance", max_iter=100,
                    isreal=True, verbose=True
        )
    >>> S_sobi = sobi.fit_transform(X)  # source signals
    >>> A = sobi.mixing_  # mixing matrix

    >>> # plot results

    >>> mplt.figure(figsize=(12,10))
    >>> models = [X, S, S_sobi]
    >>> names = [
    >>>     "Observations (mixed signal)",
    >>>     "True source signals",
    >>>     "SOBI recovered source signals",
    >>> ]
    >>> colors = ["red", "steelblue", "orange"]
    >>> for ii, (model, name) in enumerate(zip(models, names), 1):
    >>>     mplt.subplot(5, 1, ii)
    >>>     mplt.title(name)
    >>>     for sig, color in zip(model.T, colors):
    >>>         mplt.plot(sig, color=color)
    >>> mplt.tight_layout()
    >>> mplt.show()

    >>> # check inverse transform (recover X from estimated S)
    >>> np.allclose(sobi.inverse_transform(S_sobi), X, atol=5e-1)

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
        """Initialize the SOBI instance."""
        self.n_components = n_components
        self.n_lags = n_lags
        self.tol = tol
        self.max_iter = max_iter
        self.whiten = whiten
        self.isreal = isreal
        self.verbose = verbose

        self.components_ = None
        self.mean_ = None
        self.mixing_ = None
        self.whitening_ = None
        self.covs = None
        self.covs_diag = None

    def fit_transform(self, X):
        """Fit the model and recover the sources from X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input mixed signals.

        Returns
        -------
        S : ndarray of shape (n_samples, n_components)
            Estimated source signals.
        """
        return self._fit_transform(X, compute_sources=True)

    def fit(self, X):
        """Fit the model to X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input mixed signals.
        """
        self._fit_transform(X, compute_sources=False)
        return self

    def transform(self, X):
        """Apply the unmixing matrix to new data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        S : ndarray of shape (n_samples, n_components)
        """
        if self.components_ is None:
            raise RuntimeError(
                "Model must be fitted before calling transform."
            )
        return _np.dot(X, self.components_.T)

    def inverse_transform(self, X):
        """Reconstruct original signals from sources.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_components)

        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_features)
        """
        if self.mixing_ is None:
            raise RuntimeError(
                "Model must be fitted before calling inverse_transform."
            )
        return _np.dot(X, self.mixing_.T) + self.mean_[None, :]

    def _fit_transform(self, X, compute_sources=False):
        n_samples, n_features = X.shape
        rank = min(n_samples, n_features)
        n_components = self.n_components

        if n_components is None or n_components > rank:
            n_components = rank
            warnings.warn(
                "n_components is None or greater than data rank. " /
                + "Using min(X.shape)"
            )

        XT, K, X_mean = self._whiten_data(X.T, n_components)
        covs = self._calc_covariances(XT)
        covs = covs + 1j * _np.zeros_like(covs)

        W, covs_diag = self._joint_diag(
            covs, tol=self.tol, max_iter=self.max_iter
        )

        if self.isreal:
            W = _np.real(W)
            covs_diag = _np.real(covs_diag)
            covs = _np.real(covs)

        if compute_sources:
            S = _np.dot(W.T, XT).T
        else:
            S = None

        if self.whiten == "unit-variance" and S is not None:
            S_std = S.std(axis=0, keepdims=True)
            S /= S_std
            W /= S_std.T

        self.components_ = _np.dot(W.T, K)
        self.mean_ = X_mean
        self.whitening_ = K
        self.mixing_ = _np.linalg.pinv(self.components_)
        self.covs = covs
        self.covs_diag = covs_diag

        return S

    def _whiten_data(self, X, n_components=None):
        """Compress & whiten data.

        Projects into the n_components-dimensional space
        of principal components z = S^-1 U^T X, where S and U are such that
        X = U S V.T . Assumes data vectors as rows.

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
        Xnm, X_mean = self._center_data(X)
        U, s, _ = _np.linalg.svd(Xnm, full_matrices=False)
        K = (U / s).T[:n_components]
        X_white = K @ X  # z = U.T S^-1 X

        return X_white, K, X_mean

    def _center_data(self, X):
        """Center data about its mean.

        Args:
            X ((m,n)-array): input data, data vectors (samples) as rows.

        Returns:
            Xnm ((m,n)-array): mean-subtracted data matrix.
            X_mean (m-array): mean value for each sample.
        """
        X_ = X.copy()
        X_mean = X_.mean(axis=-1)
        X_centered = X_ - X_mean[:, None]
        return X_centered, X_mean

    def _calc_covariances(self, X, n_lags=None):
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

        timelagged_covs = _np.empty((lags + 1, m, m))
        X_0, _ = self._center_data(X[:, :n_])

        for k in range(lags + 1):
            X_k, _ = self._center_data(X[: , k:k + n_])
            cov = X_0 @ X_k.T / n_
            timelagged_covs[k] = cov

        return timelagged_covs

    def _off_diagonal_frobenius(self, M):
        """Compute the sum of squared off-diag entries of a matrix.

        Args:
            M (m,n-array): input matrix.

        Returns:
            off(M) (float): the sum of squares of the off-diagonal entries of
            M.
        """
        M_ = M.copy()
        M_ -= _np.diag(_np.diag(M_))
        return _np.linalg.norm(M_, ord='fro')**2

    def _jacobi_rotation(self, M):
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
        h = _np.array([
            M[:, 0, 0] - M[:, 1, 1],
            M[:, 0, 1] + M[:, 1, 0],
            1j*(M[:, 1, 0] - M[:, 0, 1])
        ]).T  # ref [2], eq. (5)

        G = _np.real(h.T @ h)  # ref [2], eq. (4)
        _, eigenvecs = _np.linalg.eigh(G)
        x, y, z = eigenvecs[:, -1]

        if x < 0:
            x, y, z = -x, -y, -z  # inner-rotations remark on [2]

        r = _np.linalg.norm([x, y, z])  # [2], eqs (6)
        c = _np.sqrt((x + r) / 2 / r)
        s = (y - 1j * z) / _np.sqrt(2 * r * (x + r))

        R_ijcs = _np.array([
            [c, _np.conj(s)],
            [-s, _np.conj(c)]
        ])  # [2], eq (2)

        return R_ijcs

    def _joint_diag(
        self, matrices, tol, max_iter, verbose=False
    ):
        """Joint-diagonalize a set of symmetric, square matrices.

        Does so by iteratively applying Jacobi rotations, as described in
        Cardoso & Souloumiac. Jacobi Angles For Simultaneous Diagonalization.
        SIAM Journal on Matrix Analysis and Applications.
        https://www.researchgate.net/publication/277295728_Jacobi_Angles_For_Simultaneous_Diagonalization

        Args:
            matrices ((l, m, m)-array): the l m x m matrices to be
            jointly-diagonalized.

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
        # TODO: real-valued matrices simplify calculations
        # use of self.is_real flag is supposed to account for that
        # not implemented yet
        nr_rows = matrices.shape[1]
        ij_pairs = list(_combinations(range(nr_rows), 2))
        identity = _np.eye(nr_rows, dtype=complex)
        W = identity.copy()

        obj = _np.sum([self._off_diagonal_frobenius(m) for m in matrices])

        diff = _np.inf
        count = 0

        if verbose:
            print("Joint-diagonalizing...")
            print(f"iter {count:3d} \t objective: {obj:.2g}")

        while diff > tol and count < max_iter:
            count += 1
            for i, j in ij_pairs:
                submat_idcs = _np.ix_([i, j], [i, j])
                idx = (slice(None), ) + submat_idcs
                R = self._jacobi_rotation(matrices[idx])
                G = identity.copy()
                G[[i, j]][:, [i, j]] = R

                W = W @ G.T
                matrices = _np.matmul(_np.matmul(G, matrices), G.T)

            new_obj = _np.sum([
                self._off_diagonal_frobenius(m) for m in matrices
            ])
            diff = _np.abs(obj - new_obj)
            obj = new_obj

            if verbose:
                print(f"iter {count:3d} \t objective: {obj:.2g}")

        return W, matrices
