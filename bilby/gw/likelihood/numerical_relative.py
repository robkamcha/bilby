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

    def setup_bins_iteration(self):
        pass

    def setup_bins(self, N_target_bins):

        """
        Constructs a sparse frequency grid
        starting from the uniform frequency grid.
        The uniform frequency will be bisected until
        the error crosses the threshold value.
        """
        # FIXME
        catch_errors = None
        proposed_bins = self.setup_bins_iteration()

        if len(proposed_bins) != N_target_bins:
            logger.info(
                f"Rerunning the binning algorithm with {len(proposed_bins)} proposed bins"
                f"and {len(N_target_bins)} target bins"
            )

            self.setup_bins(len(proposed_bins))

        elif len(proposed_bins) < self.minimum_bins:
            self.generate_perturbed_parameters()
            logger.info(
                f"The previous set of pertrubed parameters resulted in "
                f"too few bins. Updated perturbed chirp mass to "
                f"{self.perturbed_parameters['chirp_mass']}"
                f"and mass ratio to {self.perturbed_parameters['mass_ratio']}."
            )
            # FIXME: Update pertrubed_strains
            self.setup_bins(self.minimum_bins)

        else:
            sum_errors = np.sum(np.array(catch_errors))
            logger.info(f"Observed errors: {sum_errors}")
            logger.info(f"Expected errors: {self.delta}")
            logger.info(
                f"Terminating the bisection algorithm. "
                f"The number of bins made: {len(proposed_bins)}"
            )

            # FIXME
            accepted_grid = np.insert(proposed_bins, 0, self.band_indices[0])
            accepted_grid = sorted(np.unique(accepted_grid))

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

    def find_nearest_index(self, frequency_value, frequency_array):
        """
        Method to find the index in the `frequency_array` that is
        nearest to the `frequency_value`
        """
        index = np.argmin(np.abs(frequency_array - frequency_value))
        return index

    def compute_likelihood_error(self, minimum_frequency, maximum_frequency, relative=False):
        """
        Function to compute the errors between
        the exact likelihood and the
        relative binning (or approximate) likelihood

        minimum_frequency: Lower bound used for likelihood integration
        maximum_frequency: Upper bound used for likelihood integration
        """
        likelihood_error = False
        return likelihood_error
