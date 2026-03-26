import numpy as np
from .base import GravitationalWaveTransient
from ...core.utils import logger
# from scipy.optimize import differential_evolution
# from ..utils import noise_weighted_inner_product
from ...core.likelihood import _fallback_to_parameters

"""
Remaining things to do
1. Method to generate waveform on a desired frequency grid
2. Summary data computation between a certain fmin and fmax
3. Method to compute the exact likelihood
4. I need a way to call detector frame waveform at a given frequency value
"""


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
        self.uniform_frequency_array_resolution = self.frequency_array[1] - \
            self.frequency_array[0]
        # Assign fiducial parameters and use it to generate perturbed parameters

        self.fiducial_parameters = fiducial_parameters
        self.fiducial_detector_response = self.compute_full_detector_response(
            fiducial_parameters)
        self.maximum_frequency = self.compute_last_nonzero_frequency(
            self.fiducial_detector_response)
        self.inverse_psd = self.compute_inverse_psd()

        self.perturbed_parameters = self.generate_perturbed_parameters()
        self.perturbed_detector_response = self.compute_full_detector_response(
            self.perturbed_parameters)

        # Starting value of the iteration is self.minimum_bins
        self.relative_binning_frequency_index = self.setup_bins(self.minimum_bins,
                                                                self.fiducial_detector_response,
                                                                self.perturbed_detector_response)

        self.relative_binning_frequency_array = self.frequency_array[
            self.relative_binning_frequency_index]
        self.relative_binning_centers = 0.5 * \
            (self.relative_binning_frequency_array[1:] +
             self.relative_binning_frequency_array[:-1])
        self.relative_binning_sizes = np.diff(
            self.relative_binning_frequency_index)
        self.relative_binning_sizes[-1] += 1
        self.relative_binning_widths = np.diff(
            self.relative_binning_frequency_array)
        self.per_detector_fiducial_waveform_points = self.compute_per_detector_fiducial_waveform_points()

        # After setting up sparse binning grid, I will update the waveform_generator to
        self.waveform_generator.waveform_arguments["frequency_bin_edges"] = self.relative_binning_frequency_array

    def compute_inverse_psd(self):
        """
        Returns a dictionary of inverse power spectral densities
        """
        inverse_psd = {}
        for interferometer in self.interferometers:
            name = interferometer.name
            _temp_psd = np.power(
                interferometer.power_spectral_density_array, -1)
            _temp_psd[np.isnan(_temp_psd)] = 0.0
            _temp_psd[np.isinf(_temp_psd)] = 0.0
            inverse_psd[name] = _temp_psd
        return inverse_psd

    def compute_per_detector_fiducial_waveform_points(self):
        per_detector_fiducial_waveform_points = dict()
        for interferometer in self.interferometers:
            name = interferometer.name
            per_detector_fiducial_waveform_points[name] = (
                self.fiducial_detector_response[name][self.relative_binning_frequency_index]
            )
        return per_detector_fiducial_waveform_points

    def compute_full_detector_response(self, parameters):
        fiducial_polarizations = self.waveform_generator.frequency_domain_strain(
            parameters)
        fiducial_detector_response = {}
        for interferometer in self.interferometers:
            name = interferometer.name
            fiducial_detector_response[name] = interferometer.get_detector_response(fiducial_polarizations,
                                                                                    parameters)
        return fiducial_detector_response

    def compute_last_nonzero_frequency(self, detector_response):
        """
        detector_response is a dictionary with {'name': 1D array of complex numbers}
        """
        last_nonzero_indices = []
        for _, value in detector_response.items():
            if value[-1] != 0j:
                last_nonzero_indices.append(-1)
                continue
            else:
                nonzero_indices = np.where(value[::-1] != 0j)[0]
                last_nonzero_index = len(value) - 1 - nonzero_indices[0]
                last_nonzero_indices.append(last_nonzero_index)

        last_nonzero_frequency = self.waveform_generator.frequency_array[min(
            last_nonzero_indices)]
        return last_nonzero_frequency

    def setup_bins(self, N_target_bins, fiducial_detector_response, perturbed_detector_response):
        """
        Constructs a sparse frequency grid
        starting from the uniform frequency grid.
        The uniform frequency will be bisected until
        the error crosses the threshold value.
        """

        catch_errors = []

        proposed_frequency_index = self.setup_bins_using_bisection_method(N_target_bins,
                                                                          fiducial_detector_response,
                                                                          perturbed_detector_response,
                                                                          self.minimum_frequency,
                                                                          self.maximum_frequency,
                                                                          catch_errors)

        counter = 0
        if N_target_bins != len(proposed_frequency_index):

            logger.info(
                f"Rerunning the binning algorithm with {len(proposed_frequency_index)} proposed bins"
                f"and {len(N_target_bins)} target bins"
            )
            counter += 1
            self.setup_bins(len(proposed_frequency_index),
                            fiducial_detector_response, perturbed_detector_response)

        elif len(proposed_frequency_index) < self.minimum_bins:
            perturbed_parameters = self.generate_perturbed_parameters()
            perturbed_detector_response = self.compute_full_detector_response(
                perturbed_parameters)
            logger.info(
                f"The previous set of perturbed parameters resulted in "
                f"too few bins. Updated perturbed chirp mass to "
                f"{perturbed_parameters['chirp_mass']}"
                f"and mass ratio to {self.perturbed_parameters['mass_ratio']}."
            )
            # FIXME: Update perturbed_strains

            self.setup_bins(
                self.minimum_bins, fiducial_detector_response, perturbed_detector_response)

        else:
            sum_errors = np.sum(np.array(catch_errors))
            logger.info(f"Observed errors: {sum_errors}")
            logger.info(f"Expected errors: {self.delta}")
            logger.info(
                f"Terminating the bisection algorithm. "
                f"The number of bins made: {len(proposed_frequency_index)}"
            )

            # FIXME
            global_minimum_frequency_index = self.find_nearest_index(
                self.minimum_frequency, self.frequency_array)
            accepted_frequency_index = np.insert(
                proposed_frequency_index, 0, global_minimum_frequency_index)
            accepted_frequency_index = sorted(
                np.unique(accepted_frequency_index))
            return accepted_frequency_index

    def setup_bins_using_bisection_method(self, N_bins, fiducial_detector_response, perturbed_detector_response,
                                          minimum_frequency,
                                          maximum_frequency,
                                          catch_errors):

        minimum_frequency_index = self.find_nearest_index(
            minimum_frequency, self.frequency_array)
        maximum_frequency_index = self.find_nearest_index(
            maximum_frequency, self.maximum_frequency)

        allowed_error_per_bin = self.delta / N_bins
        observed_error_per_bin, observed_likelihood_per_bin = \
            self.compute_likelihood_and_error_per_bin(minimum_frequency_index,
                                                      maximum_frequency_index)

        if (maximum_frequency_index - minimum_frequency_index) <= 1:
            logger.info(
                "Cannot split this bins further. "
                "Reached uniform grid limit. "
                f"{minimum_frequency_index}... "
                f"{maximum_frequency_index}"
            )
            logger.info(
                f"Allowed error {allowed_error_per_bin}...observed error {observed_error_per_bin}"
            )
            logger.info(
                f"Apporximate likelihoods: {observed_likelihood_per_bin}\n")

            return np.array([maximum_frequency_index])

        if observed_error_per_bin < allowed_error_per_bin:
            catch_errors += [observed_error_per_bin]
            return np.array([maximum_frequency_index])

        else:
            mid_frequency_index = (
                maximum_frequency_index + minimum_frequency_index) // 2

            accepted_minimum_frequency_index = self.setup_bins_using_bisection_method(N_bins,
                                                                                      fiducial_detector_response,
                                                                                      perturbed_detector_response,
                                                                                      minimum_frequency_index,
                                                                                      mid_frequency_index,
                                                                                      catch_errors)

            accepted_maximum_frequency_index = self.setup_bins_using_bisection_method(N_bins,
                                                                                      fiducial_detector_response,
                                                                                      perturbed_detector_response,
                                                                                      mid_frequency_index,
                                                                                      maximum_frequency_index,
                                                                                      catch_errors)

            accepted_frequency_index = np.append(
                accepted_minimum_frequency_index, accepted_maximum_frequency_index)
            return accepted_frequency_index

    def compute_waveform_ratio_per_interferometer(self, waveform_polarizations, interferometer, parameters=None):
        parameters = _fallback_to_parameters(self, parameters)
        name = interferometer.name
        strain = interferometer.get_detector_response(
            waveform_polarizations=waveform_polarizations,
            parameters=parameters,
            frequencies=self.frequency_array,
        )
        reference_strain = self.per_detector_fiducial_waveform_points[name]
        waveform_ratio = strain / reference_strain

        r0 = (waveform_ratio[1:] + waveform_ratio[:-1]) / 2
        r1 = (waveform_ratio[1:] - waveform_ratio[:-1]) / \
            self.relative_binning_widths

        return [r0, r1]

    def _compute_full_waveform(self, signal_polarizations, interferometer, parameters=None):
        fiducial_waveform = self.fiducial_detector_response[interferometer.name]
        r0, r1 = self.compute_waveform_ratio_per_interferometer(
            waveform_polarizations=signal_polarizations,
            interferometer=interferometer,
            parameters=parameters
        )

        idxs = slice(
            self.relative_binning_frequency_index[0], self.relative_binning_frequency_index[-1] + 1)
        duplicated_r0 = np.repeat(r0, self.relative_binning_sizes)
        duplicated_r1 = np.repeat(r1, self.relative_binning_sizes)
        duplicated_fm = np.repeat(
            self.relative_binning_centers, self.relative_binning_sizes)

        f = interferometer.frequency_array
        full_waveform_ratio = np.zeros(f.shape[0], dtype=complex)
        full_waveform_ratio[idxs] = duplicated_r0 + \
            duplicated_r1 * (f[idxs] - duplicated_fm)
        return fiducial_waveform * full_waveform_ratio

    def calculate_snrs(self, waveform_polarizations, interferometer, return_array=True, parameters=None):
        r0, r1 = self.compute_waveform_ratio_per_interferometer(
            waveform_polarizations=waveform_polarizations,
            interferometer=interferometer,
            parameters=parameters,
        )
        a0, a1, b0, b1 = self.summary_data[interferometer.name]
        d_inner_h = np.sum(a0 * np.conjugate(r0) + a1 * np.conjugate(r1))
        h_inner_h = np.sum(b0 * np.abs(r0) ** 2 + 2 * b1 *
                           np.real(r0 * np.conjugate(r1)))
        optimal_snr_squared = h_inner_h
        complex_matched_filter_snr = d_inner_h / (optimal_snr_squared ** 0.5)

        if return_array and self.time_marginalization:
            full_waveform = self._compute_full_waveform(
                signal_polarizations=waveform_polarizations,
                interferometer=interferometer,
                parameters=parameters,
            )
            d_inner_h_array = 4 / self.waveform_generator.duration * np.fft.fft(
                full_waveform[0:-1]
                * interferometer.frequency_domain_strain.conjugate()[0:-1]
                / interferometer.power_spectral_density_array[0:-1])

        else:
            d_inner_h_array = None

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h,
            optimal_snr_squared=optimal_snr_squared.real,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_array=d_inner_h_array
        )

    def generate_perturbed_parameters(self):
        perturbed_parameters = self.fiducial_parameters.copy()

        # FIXME: In future, we may want to use a fisher matrix to estimate these numbers
        chirp_mass_perturbation_percentage = 1
        mass_ratio_perturbation_percentage = 1

        chirp_mass_perturbation = 1 + \
            (1e-2 * chirp_mass_perturbation_percentage *
             np.random.uniform(-1, 1, 1)[0])
        mass_ratio_perturbation = 1 + \
            (1e-2 * mass_ratio_perturbation_percentage *
             np.random.uniform(-1, 1, 1)[0])

        perturbed_parameters['chirp_mass'] *= chirp_mass_perturbation
        perturbed_parameters['mass_ratio'] *= mass_ratio_perturbation

        logger.info(
            f"Perturbed chirp mass and mass ratio: "
            f"{perturbed_parameters['chirp_mass']} "
            f"{perturbed_parameters['mass_ratio']}"
        )
        return perturbed_parameters

    def compute_summary_data(self, frequency_array_index):
        """
        Function to compute summary data
        Value of the frequency in Hz.
        """
        A_0 = {}
        A_1 = {}
        B_0 = {}
        B_1 = {}
        bin_frequencies = self.waveform_generator.frequency_array[frequency_array_index]
        central_frequency = 0.5 * (bin_frequencies[1:] + bin_frequencies[:-1])
        for interferometer in self.interferometers:
            name = interferometer.name
            A_0[name] = np.zeros(len(frequency_array_index) - 1, dtype=complex)
            A_1[name] = np.zeros(len(frequency_array_index) - 1, dtype=complex)
            B_0[name] = np.zeros(len(frequency_array_index) - 1, dtype=complex)
            B_1[name] = np.zeros(len(frequency_array_index) - 1, dtype=complex)
            for bin_idx in range(len(frequency_array_index) - 1):
                idx_low = frequency_array_index[bin_idx]
                idx_high = frequency_array_index[bin_idx + 1]
                frequency_slice = self.waveform_generator.frequency_array[idx_low:idx_high]
                A_0[name][bin_idx] = np.sum(
                    4.0
                    / self.waveform_generator.duration
                    * interferometer.frequency_domain_strain[idx_low:idx_high]
                    * np.conj(self.fiducial_detector_response[name][idx_low:idx_high])
                    * self.inverse_psd[name][idx_low:idx_high]
                )
                A_1[name][bin_idx] = np.sum(
                    4.0
                    / self.waveform_generator.duration
                    * interferometer.frequency_domain_strain[idx_low:idx_high]
                    * np.conj(self.fiducial_detector_response[name][idx_low:idx_high])
                    * self.inverse_psd[name][idx_low:idx_high]
                    * (frequency_slice - central_frequency[bin_idx])
                )
                B_0[name][bin_idx] = np.sum(
                    4.0
                    / self.waveform_generator.duration
                    * self.fiducial_detector_response[name][idx_low:idx_high]
                    * np.conj(self.fiducial_detector_response[name][idx_low:idx_high])
                    * self.inverse_psd[name][idx_low:idx_high]
                )
                B_1[name][bin_idx] = np.sum(
                    4.0
                    / self.waveform_generator.duration
                    * self.fiducial_detector_response[name][idx_low:idx_high]
                    * np.conj(self.fiducial_detector_response[name][idx_low:idx_high])
                    * self.inverse_psd[name][idx_low:idx_high]
                    * (frequency_slice - central_frequency[bin_idx])
                )

        return A_0, A_1, B_0, B_1

    def find_nearest_index(self, frequency_value, frequency_array):
        """
        Method to find the index in the `frequency_array` that is
        nearest to the `frequency_value`
        """
        index = np.argmin(np.abs(frequency_array - frequency_value))
        return index

    def compute_likelihood_and_error_per_bin(self, minimum_frequency, maximum_frequency, relative=False):
        """
        Function to compute the errors between
        the exact likelihood and the
        relative binning (or approximate) likelihood

        minimum_frequency: Lower bound used for likelihood integration
        maximum_frequency: Upper bound used for likelihood integration
        """
        likelihood_error = False
        return likelihood_error
