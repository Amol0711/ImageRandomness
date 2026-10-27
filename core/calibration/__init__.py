"""Finite-reference calibration for image statistics.

All image statistics use fractions, not percentage units. Moment identities are
exact under the specified independent pixel laws. Gaussian intervals are normal
approximations and are not finite-sample guarantees.
"""

from .core import (
    BACIMoments, EdgeMoments, Interval, PairConstants, UACIMoments,
    baci_from_images, baci_statistic, block_entropy, count_entropy,
    fixed_baci_moments, fixed_difference_law,
    fixed_uaci_moments, gaussian_interval, grid_edge_weights, grid_coefficients,
    iid_baci_moments, iid_pair_constants, known_null_rank_pvalue,
    local_entropy_interval, local_entropy_mean, local_entropy_from_counts,
    npcr_statistic, triangular_difference_law, uaci_statistic, weighted_edge_moments,
)

__all__ = [
    'BACIMoments', 'EdgeMoments', 'Interval', 'PairConstants', 'UACIMoments',
    'baci_from_images', 'baci_statistic', 'block_entropy', 'count_entropy', 'fixed_baci_moments',
    'fixed_difference_law', 'fixed_uaci_moments', 'gaussian_interval',
    'grid_edge_weights', 'grid_coefficients', 'iid_baci_moments',
    'iid_pair_constants', 'known_null_rank_pvalue', 'local_entropy_interval',
    'local_entropy_mean', 'local_entropy_from_counts', 'npcr_statistic',
    'triangular_difference_law', 'uaci_statistic', 'weighted_edge_moments',
]
__version__ = '0.1.0'
