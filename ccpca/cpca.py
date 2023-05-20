#  Because pybind11 cannot generate default parameters well, this code is to set them
import cpca_cpp


class CPCA(cpca_cpp.CPCA):
    """Contrastive PCA with efficient C++ implemetation with Eigen.

    Parameters
    ----------
    n_components: int, optional, (default=2)
        A number of componentes to take.
    standardize: boo, optional, (default=True)
        Whether standardize input matrices or not.
    Attributes
    ----------
    None.
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from sklearn import datasets

    >>> from cpca import CPCA

    >>> dataset = datasets.load_iris()
    >>> X = dataset.data
    >>> y = dataset.target

    >>> # apply cPCA
    >>> cpca = CPCA()
    >>> X_new = cpca.fit_transform(fg=X, bg=X[y != 0], alpha=0.84)

    >>> # plot figures
    >>> plt.figure()
    >>> colors = ['navy', 'turquoise', 'darkorange']
    >>> lw = 2
    >>> for color, i, target_name in zip(colors, [0, 1, 2], [0, 1, 2]):
    ...     plt.scatter(
    ...         X_new[y == i, 0],
    ...         X_new[y == i, 1],
    ...         color=color,
    ...         alpha=.8,
    ...         lw=lw,
    ...         label=target_name)
    >>> plt.legend(loc='best', shadow=False, scatterpoints=1)
    >>> plt.title('cPCA of IRIS dataset with alpha=0.84')
    >>> plt.show()
    Notes
    -----
    """

    def __init__(self, n_components=2, standardize=True):
        super().__init__(n_components, standardize)

    def initialize(self):
        """Reset components obtained by fit()
        """
        super().initialize()

    def fit(self,
            fg,
            bg,
            auto_alpha_selection=True,
            alpha=None,
            eta=1e-3,
            convergence_ratio=1e-2,
            max_iter=10,
            keep_reports=False):
        """Fit the model with a foreground matrix and a background matrix.

        Parameters
        ----------
        fg: array-like, shape (n_samples, n_features)
            A foreground (or target) dataset.
        bg: array-like, shape (n_samples, n_features)
            A background dataset. This column size must be the same size with
            fg. (A row size can be different from fg.)
        auto_alpha_selection:
            If True, find auto_alpha_selection for fit. Otherwise, compute PCs
            based on input alpha.
        eta: float, optional, (default=1e-3)
            Small constant value that will add to covariance matrix of bg when
            applying automatic alpha selection. Smaller eta tends to allow
            a larger alpha as the best alpha.
        convergence_ratio: float, optional, (default=1e-2)
            Threshold of improvement ratio for convergence of automatic alpha
            selection.
        max_iter=10: int, optional, (default=10)
            The number of alpha updates at most.
        keep_reports: bool, optional, (default=False)
            If True, while automatic alpha selection, reports are recorded. The
            reports are the history of "alpha" values.
        Returns
        -------
        self.
        """
        if alpha == None:
            alpha = 0.0
            auto_alpha_selection = True
        else:
            auto_alpha_selection = False

        super().fit(fg, bg, auto_alpha_selection, alpha, eta,
                    convergence_ratio, max_iter, keep_reports)

        return self

    def transform(self, X):
        """Obtaining transformed result Y with X and current PCs.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Testing data, where n_samples is the number of samples and
            n_features is the number of features. n_features must be the same
            size with the traiding data's features used for partial_fit.
        Returns
        -------
        Y : array-like, shape (n_samples, n_components)
            The transformed (or projected) result.
        """
        return super().transform(X)

    def fit_transform(self,
                      fg,
                      bg,
                      auto_alpha_selection=True,
                      alpha=None,
                      eta=1e-3,
                      convergence_ratio=1e-2,
                      max_iter=10,
                      keep_reports=False):
        """Fit the model with a foreground matrix and a background matrix and
        then obtain transformed result of fg and current PCs.

        Parameters
        ----------
        fg: array-like, shape (n_samples, n_features)
            A foreground (or target) dataset.
        bg: array-like, shape (n_samples, n_features)
            A background dataset. This column size must be the same size with
            fg. (A row size can be different from fg.)
        auto_alpha_selection:
            If True, find auto_alpha_selection for fit. Otherwise, compute PCs
            based on input alpha.
        alpha: float
            A contrast parameter, which quantifies the trade-off between having
            high target variance and low background variance. alpha must be
            equal to or larger than 0. If 0, the result will be the same with
            the ordinary PCA. If auto_alpha_selection is True, this alpha is
            used as an initial alpha value for auto selection.
        eta: float, optional, (default=1e-3)
            Small constant value that will add to covariance matrix of bg when
            applying automatic alpha selection. Smaller eta tends to allow
            a larger alpha as the best alpha.
        convergence_ratio: float, optional, (default=1e-2)
            Threshold of improvement ratio for convergence of automatic alpha
            selection.
        max_iter=10: int, optional, (default=10)
            The number of alpha updates at most.
        keep_reports: bool, optional, (default=False)
            If True, while automatic alpha selection, reports are recorded. The
            reports are the history of "alpha" values.
        Returns
        -------
        Y : array-like, shape (n_samples, n_components)
            The transformed (or projected) result.
        """
        self.fit(fg, bg, auto_alpha_selection, alpha, eta, convergence_ratio,
                 max_iter, keep_reports)
        return self.transform(fg)

    def update_components(self, alpha):
        """Update the components with a new contrast parameter alpha. Before
        using this, at least one time, fit() or fit_transform() must be called
        to build an initial result of the components.

        Parameters
        ----------
        alpha: float
            A contrast parameter, which quantifies the trade-off between having
            high target variance and low background variance. alpha must be
            equal to or larger than 0. If 0, the result will be the same with
            the ordinary PCA.
        Returns
        -------
        None.
        """
        super().update_components(alpha)

    def best_alpha(self,
                   fg,
                   bg,
                   init_alpha=0.0,
                   eta=1e-3,
                   convergence_ratio=1e-2,
                   max_iter=10,
                   keep_reports=False):
        """Finds the best contrast parameter alpha which has high discrepancy
        score between the dimensionality reduced K and the dimensionality
        reduced R while keeping the variance of K with the ratio threshold
        var_thres_ratio.
        For cPCA, a matrix E concatenating K and R will be used as a foreground
        dataset and R will be used as a background dataset.
        Parameters
        ----------
        fg: array-like, shape (n_samples, n_features)
            A foreground (or target) dataset.
        bg: array-like, shape (n_samples, n_features)
            A background dataset. This column size must be the same size with
            fg. (A row size can be different from fg.)
        init_alpha: float, optional, (default=0.0)
            An initial value of alpha.
        epoch: int, optional, (default=10)
            The number of alpha updates.
        keep_reports: bool, optional, (default=False)
            If True, while automatic alpha selection, reports are recorded. The
            reports are the history of "alpha" values.
        Returns
        -------
        best_alpha: float
            The found best alpha.
        """
        super().best_alpha(fg,
                           bg,
                           init_alpha=init_alpha,
                           eta=eta,
                           convergence_ratio=convergence_ratio,
                           max_iter=max_iter,
                           keep_reports=keep_reports)

    def logspace(self, start, end, num, base=10.0):
        """Generate logarithmic space.

        Parameters
        ----------
        start: float
            base ** start is the starting value of the sequence.
        end: float
            base ** end is the ending value of the sequence.
        num: int
            Number of samples to generate.
        base : float, optional, (default=10.0)
            The base of the log space. The step size between the elements in
            ln(samples) / ln(base) (or log_base(samples)) is uniform.
        Returns
        -------
        samples : ndarray
            Num samples, equally spaced on a log scale.
        None.
        """
        return super().logspace(start, end, num, base)

    def get_components(self):
        """Returns current components.

        Parameters
        ----------
        None.
        Returns
        -------
        components: array-like, shape(n_features, n_components)
            Contrastive principal components.
        """
        return super().get_components()

    def get_component(self, index):
        """Returns i-th component.

        Parameters
        ----------
        index: int
            Indicates i-th component.
        Returns
        -------
        component: array-like, shape(1, n_components)
            i-th contrastive principal component.
        """
        return super().get_component(index)

    def get_eigenvalues(self):
        """Returns current eigenvalues.

        Parameters
        ----------
        None.
        Returns
        -------
        eigenvalues: array-like, shape(, n_components)
            Contrastive principal components' eigenvalues.
        """
        return super().get_eigenvalues()

    def get_eigenvalue(self, index):
        """Returns i-th eigenvalue.

        Parameters
        ----------
        index: int
            Indicates i-th eigenvalue.
        Returns
        -------
        eigenvalue: float
            i-th eigenvalue.
        """
        return super().get_eigenvalue(index)

    def get_total_pos_eigenvalue(self):
        """Returns the total of n_features positive eigenvalues (not n_components).
        This value can be used to compute the explained ratio of variance of the matrix C.

        Parameters
        ----------
        None
        Returns
        -------
        total_pos_eigenvalue: float
            The total of positive eigenvalues.
        """
        return super().get_total_pos_eigenvalue()

    def get_loadings(self):
        """Returns current principal component loadings.

        Parameters
        ----------
        None.
        Returns
        -------
        loadings: array-like, shape(n_features, n_components)
            Contrastive principal component loadings.
        """
        return super().get_loadings()

    def get_loading(self, index):
        """Returns i-th principal component loading.

        Parameters
        ----------
        index: int
            Indicates i-th principal component loading.
        Returns
        -------
        loading: array-like, shape(1, n_components)
            i-th principal component loading.
        """
        return super().get_loading(index)

    def get_reports(self):
        """Returns the reports kept while automatic selection of alpha. To get
        reports, you need to set keep_reports=True in the corresponding method.
        Parameters
        ----------
        None
        -------
        reports: array-like(n_alphas, 1),
            alpha values (at the same time optimization scores) obtained through
            best alpha selection".
        """
        return super().get_reports()
