import numpy as np
from .base import GravitationalWaveTransient
from ...core.utils import logger
# from scipy.optimize import differential_evolution
# from ..utils import noise_weighted_inner_product


class NumericalRelativeBinningGravitationalWaveTransient(GravitationalWaveTransient):
    """
    A gravitational-wave transient likelihood object.
    - This class implements a numerical relative binning scheme in contrast to the analytic one (in relative.py).
    - This class is specifically developed to perform
    parameter estimation on long duration signal including earth rotation.

    fiducial_parameters: Set of parameters used to construct the sparse grid
    minimum_bins: The bin construction method will keep running until this number is reached
    """

    def __init__(
            self, interferometers, waveform_generator,
            fiducial_parameters=None,
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

        self.minimum_bins = minimum_bins
        # Assign fiducial parameters and use it to generate pertrubed parameters
        self.fiducial_parameters = fiducial_parameters
        self.generate_perturbed_parameters()

    def setup_bins():
        pass

    def generate_perturbed_parameters(self):
        perturbed_parameters = self.fiducial_parameters.copy()

        # FIXME: In future, we may want to use a fisher matrix to estimate these numbers
        chirp_mass_perturbation_percentage = 1
        mass_ratio_perturbation_percentage = 1

        chirp_mass_perturbation = 1 + (1e-2 * chirp_mass_perturbation_percentage * np.random.uniform(-1, 1, 1)[0])
        mass_ratio_perturbation = 1 + (1e-2 * mass_ratio_perturbation_percentage * np.random.uniform(-1, 1, 1)[0])

        perturbed_parameters['chirp_mass'] *= chirp_mass_perturbation
        perturbed_parameters['mass_ratio'] *= mass_ratio_perturbation

        logger.info(
            f"Perturbed chirp mass and mass ratio: "
            f"{perturbed_parameters['chirp_mass']} "
            f"{perturbed_parameters['mass_ratio']}"
        )
        self.perturbed_parameters = perturbed_parameters

    def compute_summary_data(self, frequency_bins):
        """
        Function to compute summary data
        """
        summary_data = dict
        self.summary_data = summary_data
