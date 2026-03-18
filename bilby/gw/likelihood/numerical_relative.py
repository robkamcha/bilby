import numpy as np
from .base import GravitationalWaveTransient
from ...core.utils import logger
from scipy.optimize import differential_evolution
from ..utils import noise_weighted_inner_product




class NumericalRelativeBinningGravitationalWaveTransient(GravitationalWaveTransient):
    """
    A gravitational-wave transient likelihood object.
    - This class implements a numerical relative binning scheme in contrast to the analytic one (in relative.py).
    - This class is specifically developed to perform parameter estimation on long duration signal including earth rotation. 
    """

    def __init__(
            self, interferometers, waveform_generator, fiducial_parameters=None, 
            time_marginalization=False,
            distance_marginalization=False, phase_marginalization=False, priors=None,
            distance_marginalization_lookup_table=None,
            jitter_time=True, reference_frame="sky",
            time_reference="geocenter", earth_rotation=False,
            minimum_bins=1000,
            ):
        
        super().__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            priors=priors,
            distance_marginalization=distance_marginalization,
            time_marginalization=time_marginalization,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            jitter_time=jitter_time,
            reference_frame=reference_frame,
            time_reference=time_reference,
            phase_marginalization=phase_marginalization,
            earth_rotation=earth_rotation,
        )