import numpy as np
from scipy.spatial.distance import pdist, squareform

from scipy.stats import spearmanr


class MeasureRegistrator():
    """Keeps track of measurements in Measure Calculator."""
    k_independent_measures = {}
    k_dependent_measures = {}

    def register(self, is_k_dependent):
        def k_dep_fn(measure):
            self.k_dependent_measures[measure.__name__] = measure
            return measure

        def k_indep_fn(measure):
            self.k_independent_measures[measure.__name__] = measure
            return measure

        if is_k_dependent:
            return k_dep_fn
        return k_indep_fn

    def get_k_independent_measures(self):
        return self.k_independent_measures

    def get_k_dependent_measures(self):
        return self.k_dependent_measures


class MeasureCalculator():
    measures = MeasureRegistrator()

    def __init__(self, X, Z, k_max):
        self.k_max = k_max
        self.pairwise_X = squareform(pdist(X))
        self.pairwise_Z = squareform(pdist(Z))

        self.pairwise_X_norm = self.pairwise_X/self.pairwise_X.max()
        self.pairwise_Z_norm = self.pairwise_Z/self.pairwise_Z.max()

        self.neighbours_X, self.ranks_X = \
            self._neighbours_and_ranks(self.pairwise_X, k_max)
        self.neighbours_Z, self.ranks_Z = \
            self._neighbours_and_ranks(self.pairwise_Z, k_max)

        self.K_kX5, self.K_kZ5, self.K_kX_norm5, self.K_kZ_norm5, self.llrmse_X5, self.llrmse_Z5, self.llrmse_X_norm5, self.llrmse_Z_norm5 = self.Lipschitz(k = 5)

        self.K_kX15, self.K_kZ15, self.K_kX_norm15, self.K_kZ_norm15, self.llrmse_X15, self.llrmse_Z15, self.llrmse_X_norm15, self.llrmse_Z_norm15 = self.Lipschitz(k=15)

        self.K_kX15, self.K_kZ15, self.K_kX_norm15, self.K_kZ_norm15, self.llrmse_X15, self.llrmse_Z15, self.llrmse_X_norm15, self.llrmse_Z_norm15 = self.Lipschitz(
                k=5)

    @staticmethod
    def _neighbours_and_ranks(distances, k):
        """
        Inputs: 
        - distances,        distance matrix [n times n], 
        - k,                number of nearest neighbours to consider
        Returns:
        - neighbourhood,    contains the sample indices (from 0 to n-1) of kth nearest neighbor of current sample [n times k]
        - ranks,            contains the rank of each sample to each sample [n times n], whereas entry (i,j) gives the rank that sample j has to i (the how many 'closest' neighbour j is to i) 
        """
        # Warning: this is only the ordering of neighbours that we need to
        # extract neighbourhoods below. The ranking comes later!
        indices = np.argsort(distances, axis=-1, kind='stable')

        # Extract neighbourhoods.
        neighbourhood = indices[:, 1:k+1]

        # Convert this into ranks (finally)
        ranks = indices.argsort(axis=-1, kind='stable')

        return neighbourhood, ranks

    def get_X_neighbours_and_ranks(self, k):
        return self.neighbours_X[:, :k], self.ranks_X

    def get_Z_neighbours_and_ranks(self, k):
        return self.neighbours_Z[:, :k], self.ranks_Z

    def compute_k_independent_measures(self):
        return {key: fn(self) for key, fn in
                self.measures.get_k_independent_measures().items()}

    def compute_k_dependent_measures(self, k):
        return {key: fn(self, k) for key, fn in
                self.measures.get_k_dependent_measures().items()}

    def compute_measures_for_ks(self, ks):
        return {
            key: np.array([fn(self, k) for k in ks])
            for key, fn in self.measures.get_k_dependent_measures().items()
        }

    @measures.register(False)
    def stress(self):
        sum_of_squared_differences = \
            np.square(self.pairwise_X-self.pairwise_Z).sum()
        sum_of_squares = np.square(self.pairwise_Z).sum()

        return np.sqrt(sum_of_squared_differences/sum_of_squares)

    @measures.register(False)
    def rmse(self):
        n = self.pairwise_X.shape[0]
        sum_of_squared_differences = np.square(
            self.pairwise_X-self.pairwise_Z).sum()
        return np.sqrt(sum_of_squared_differences/n**2)

    @staticmethod
    def _trustworthiness(X_neighbourhood, X_ranks, Z_neighbourhood,
                         Z_ranks, n, k):
        '''
        Calculates the trustworthiness measure between the data space `X`
        and the latent space `Z`, given a neighbourhood parameter `k` for
        defining the extent of neighbourhoods.
        '''

        result = 0.0

        # Calculate number of neighbours that are in the $k$-neighbourhood
        # of the latent space but not in the $k$-neighbourhood of the data
        # space.
        for row in range(X_ranks.shape[0]):
            missing_neighbours = np.setdiff1d(
                Z_neighbourhood[row],
                X_neighbourhood[row]
            )

            for neighbour in missing_neighbours:
                result += (X_ranks[row, neighbour]-k)

        return 1-2/(n*k*(2*n-3*k-1))*result

    @measures.register(True)
    def trustworthiness(self, k):
        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)
        n = self.pairwise_X.shape[0]
        return self._trustworthiness(X_neighbourhood, X_ranks, Z_neighbourhood,
                                     Z_ranks, n, k)

    @measures.register(True)
    def continuity(self, k):
        '''
        Calculates the continuity measure between the data space `X` and the
        latent space `Z`, given a neighbourhood parameter `k` for setting up
        the extent of neighbourhoods.

        This is just the 'flipped' variant of the 'trustworthiness' measure.
        '''

        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)
        n = self.pairwise_X.shape[0]
        # Notice that the parameters have to be flipped here.
        return self._trustworthiness(Z_neighbourhood, Z_ranks, X_neighbourhood,
                                     X_ranks, n, k)

    @measures.register(True)
    def neighbourhood_loss(self, k):
        '''
        Calculates the neighbourhood loss quality measure between the data
        space `X` and the latent space `Z` for some neighbourhood size $k$
        that has to be pre-defined.
        '''

        X_neighbourhood, _ = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, _ = self.get_Z_neighbours_and_ranks(k)

        result = 0.0
        n = self.pairwise_X.shape[0]

        for row in range(n):
            shared_neighbours = np.intersect1d(
                X_neighbourhood[row],
                Z_neighbourhood[row],
                assume_unique=True
            )

            result += len(shared_neighbours)/k

        return 1.0-result/n

    @measures.register(True)
    def rank_correlation(self, k):
        '''
        Calculates the spearman rank correlation of the data
        space `X` with respect to the latent space `Z`, subject to its $k$
        nearest neighbours.
        '''

        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)

        n = self.pairwise_X.shape[0]
        # we gather
        gathered_ranks_x = []
        gathered_ranks_z = []
        for row in range(n):
            # we go from X to Z here:
            for neighbour in X_neighbourhood[row]:
                rx = X_ranks[row, neighbour]
                rz = Z_ranks[row, neighbour]
                gathered_ranks_x.append(rx)
                gathered_ranks_z.append(rz)
        rs_x = np.array(gathered_ranks_x)
        rs_z = np.array(gathered_ranks_z)
        coeff, _ = spearmanr(rs_x, rs_z)

        ##use only off-diagonal (non-trivial) ranks:
        # inds = ~np.eye(X_ranks.shape[0],dtype=bool)
        # coeff, pval = spearmanr(X_ranks[inds], Z_ranks[inds])
        return coeff

    @measures.register(True)
    def mrre(self, k):
        '''
        Calculates the mean relative rank error quality metric of the data
        space `X` with respect to the latent space `Z`, subject to its $k$
        nearest neighbours.
        '''

        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)

        n = self.pairwise_X.shape[0]

        # First component goes from the latent space to the data space, i.e.
        # the relative quality of neighbours in `Z`.

        mrre_ZX = 0.0
        for row in range(n):
            for neighbour in Z_neighbourhood[row]:
                rx = X_ranks[row, neighbour]
                rz = Z_ranks[row, neighbour]

                mrre_ZX += abs(rx-rz)/rz

        # Second component goes from the data space to the latent space,
        # i.e. the relative quality of neighbours in `X`.

        mrre_XZ = 0.0
        for row in range(n):
            # Note that this uses a different neighbourhood definition!
            for neighbour in X_neighbourhood[row]:
                rx = X_ranks[row, neighbour]
                rz = Z_ranks[row, neighbour]

                # Note that this uses a different normalisation factor
                mrre_XZ += abs(rx-rz)/rx

        # Normalisation constant
        C = n*sum([abs(2*j-n-1)/j for j in range(1, k+1)])
        return mrre_ZX/C, mrre_XZ/C

    # Get Metric K-min and K-max
    def Lipschitz(self, k=5):
        X_neighbourhood, _ = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, _ = self.get_Z_neighbours_and_ranks(k)

        disX_kX = self.pairwise_X[:, X_neighbourhood][range(self.pairwise_X.shape[0]),
               range(self.pairwise_X.shape[0]), :]
        disZ_kX = self.pairwise_Z[:, X_neighbourhood][range(self.pairwise_Z.shape[0]),
               range(self.pairwise_X.shape[0]), :]

        disX_kZ = self.pairwise_X[:, Z_neighbourhood][range(self.pairwise_X.shape[0]),
               range(self.pairwise_X.shape[0]), :]
        disZ_kZ = self.pairwise_Z[:, Z_neighbourhood][range(self.pairwise_Z.shape[0]),
               range(self.pairwise_X.shape[0]), :]

        disX_kX_norm = self.pairwise_X_norm[:, X_neighbourhood][range(self.pairwise_X.shape[0]),
                  range(self.pairwise_X.shape[0]), :]
        disZ_kX_norm = self.pairwise_Z_norm[:, X_neighbourhood][range(self.pairwise_Z.shape[0]),
                  range(self.pairwise_X.shape[0]), :]

        disX_kZ_norm = self.pairwise_X_norm[:, Z_neighbourhood][range(self.pairwise_X.shape[0]),
                  range(self.pairwise_X.shape[0]), :]
        disZ_kZ_norm = self.pairwise_Z_norm[:, Z_neighbourhood][range(self.pairwise_Z.shape[0]),
                  range(self.pairwise_X.shape[0]), :]


        K_kX = np.maximum((disX_kX/disZ_kX), (disZ_kX/disX_kX))
        K_kZ = np.maximum((disX_kZ/disZ_kZ), (disZ_kZ/disX_kZ))
        K_kX_norm = np.maximum((disX_kX_norm/disZ_kX_norm), (disZ_kX_norm/disX_kX_norm))
        K_kZ_norm = np.maximum((disX_kZ_norm/disZ_kZ_norm), (disZ_kZ_norm/disX_kZ_norm))


        # calculation of local-rmse
        nk = disX_kX.shape[0]*disX_kX.shape[1]
        llrmse_X = np.sqrt(np.square(disX_kX-disZ_kX).sum()/nk)
        llrmse_Z = np.sqrt(np.square(disX_kZ-disZ_kZ).sum()/nk)
        llrmse_X_norm = np.sqrt(np.square(disX_kX_norm-disZ_kX_norm).sum()/nk)
        llrmse_Z_norm = np.sqrt(np.square(disX_kZ_norm-disZ_kZ_norm).sum()/nk)



        return K_kX, K_kZ, K_kX_norm , K_kZ_norm, llrmse_X, llrmse_Z, llrmse_X_norm, llrmse_Z_norm


    @measures.register(False)
    def K_5X_min(self):
        return self.K_kX5.min()
    @measures.register(False)
    def K_5X_max(self):
        return self.K_kX5.max()
    @measures.register(False)
    def K_5X_avg(self):
        return self.K_kX5.mean()
    @measures.register(False)
    def K_norm_5X_min(self):
        return self.K_kX_norm5.min()
    @measures.register(False)
    def K_norm_5X_max(self):
        return self.K_kX_norm5.max()
    @measures.register(False)
    def K_norm_5X_avg(self):
        return self.K_kX_norm5.mean()
    @measures.register(False)
    def K_15X_min(self):
        return self.K_kX15.min()
    @measures.register(False)
    def K_15X_max(self):
        return self.K_kX15.max()
    @measures.register(False)
    def K_15X_avg(self):
        return self.K_kX15.mean()
    @measures.register(False)
    def K_norm_15X_min(self):
        return self.K_kX_norm15.min()
    @measures.register(False)
    def K_norm_15X_max(self):
        return self.K_kX_norm15.max()
    @measures.register(False)
    def K_norm_15X_avg(self):
        return self.K_kX_norm15.mean()

    @measures.register(False)
    def K_5Z_min(self):
        return self.K_kZ5.min()
    @measures.register(False)
    def K_5Z_max(self):
        return self.K_kZ5.max()
    @measures.register(False)
    def K_5Z_avg(self):
        return self.K_kX5.mean()
    @measures.register(False)
    def K_norm_5Z_min(self):
        return self.K_kZ_norm5.min()
    @measures.register(False)
    def K_norm_5Z_max(self):
        return self.K_kZ_norm5.max()
    @measures.register(False)
    def K_norm_5Z_avg(self):
        return self.K_kZ_norm5.mean()
    @measures.register(False)
    def K_15Z_min(self):
        return self.K_kZ15.min()
    @measures.register(False)
    def K_15Z_max(self):
        return self.K_kZ15.max()
    @measures.register(False)
    def K_15Z_avg(self):
        return self.K_kZ15.mean()
    @measures.register(False)
    def K_norm_15Z_min(self):
        return self.K_kZ_norm15.min()
    @measures.register(False)
    def K_norm_15Z_max(self):
        return self.K_kZ_norm15.max()
    @measures.register(False)
    def K_norm_15Z_avg(self):
        return self.K_kZ_norm15.mean()

    @measures.register(False)
    def llrmse_X5(self):
        return self.llrmse_X5
    @measures.register(False)
    def llrmse_Z5(self):
        return self.llrmse_Z5
    @measures.register(False)
    def llrmse_X_norm5(self):
        return self.llrmse_X_norm5
    @measures.register(False)
    def llrmse_Z_norm5(self):
        return self.llrmse_Z_norm5

    @measures.register(False)
    def llrmse_X15(self):
        return self.llrmse_X15
    @measures.register(False)
    def llrmse_Z15(self):
        return self.llrmse_Z15
    @measures.register(False)
    def llrmse_X_norm15(self):
        return self.llrmse_X_norm15
    @measures.register(False)
    def llrmse_Z_norm15(self):
        return self.llrmse_Z_norm15


    @measures.register(False)
    def density_global(self, sigma=0.1):
        X = self.pairwise_X
        X = X/X.max()
        Z = self.pairwise_Z
        Z = Z/Z.max()

        density_x = np.sum(np.exp(-(X**2)/sigma), axis=-1)
        density_x /= density_x.sum(axis=-1)

        density_z = np.sum(np.exp(-(Z**2)/sigma), axis=-1)
        density_z /= density_z.sum(axis=-1)

        return np.abs(density_x-density_z).sum()

    @measures.register(False)
    def density_kl_global(self, sigma=0.1):
        X = self.pairwise_X
        X = X/X.max()
        Z = self.pairwise_Z
        Z = Z/Z.max()

        density_x = np.sum(np.exp(-(X**2)/sigma), axis=-1)
        density_x /= density_x.sum(axis=-1)

        density_z = np.sum(np.exp(-(Z**2)/sigma), axis=-1)
        density_z /= density_z.sum(axis=-1)

        return (density_x*(np.log(density_x)-np.log(density_z))).sum()

    @measures.register(False)
    def density_kl_global_10(self):
        return self.density_kl_global(10.)

    @measures.register(False)
    def density_kl_global_1(self):
        return self.density_kl_global(1.)

    @measures.register(False)
    def density_kl_global_01(self):
        return self.density_kl_global(0.1)

    @measures.register(False)
    def density_kl_global_001(self):
        return self.density_kl_global(0.01)

    @measures.register(False)
    def density_kl_global_0001(self):
        return self.density_kl_global(0.001)

    @measures.register(False)
    def density_kl_global_00001(self):
        return self.density_kl_global(0.0001)

    @measures.register(False)
    def density_kl_global_000001(self):
        return self.density_kl_global(0.00001)

    @measures.register(False)
    def density_kl_global_0000001(self):
        return self.density_kl_global(0.000001)
