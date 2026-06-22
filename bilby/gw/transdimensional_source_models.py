"""
This file contains source models that can be used for transdimensional inference with tBilby
"""

import inspect
import numpy as np
from .source import lal_binary_black_hole, lal_binary_neutron_star

from .detector.networks import InterferometerList


# class TransdimensionalSourceModelGenerator:
#     """
#     A class to hold the factory for the transdimensional source model, and any other related functions.
#     """

#     def __init__(self, n_max: int, model_parameter_names: list[str], base_function, ifo_list=None):
#         self.n_max = n_max
#         self.model_parameter_names = model_parameter_names
#         self.ifo_list = ifo_list
#         self.base_function = base_function

#     def get_signal_model(self):
#         """
#         Get the signal model function with the correct number of components and the correct signature.
#         """

#         n_max = self.n_max
#         for i in range(n_max):


#         def signal_model(frequency_array, **kwargs):
#             n = int(kwargs['n'])
#             SNR = [kwargs[f'SNR{i}'] for i in range(n_max)]
#             f   = [kwargs[f'f{i}']   for i in range(n_max)]
#             Q   = [kwargs[f'Q{i}']   for i in range(n_max)]
#             phi = [kwargs[f'phi{i}'] for i in range(n_max)]
#             dt  = [kwargs[f'dt{i}'] for i in range(n_max)]
#             geocent_time = kwargs['geocent_time']
#             psi = kwargs['psi']
#             ra  = kwargs['ra']
#             dec = kwargs['dec']
#             e   = kwargs['e']

#             model = {
#                 "plus":  np.zeros(frequency_array.shape, dtype='complex128'),
#                 "cross": np.zeros(frequency_array.shape, dtype='complex128'),
#             }
#             for i in range(n):
#                 A = antenna_conversion_SNR_to_amplitude(
#                     SNR[i], Q[i], f[i], e, ra, dec, geocent_time + dt[i], psi, ifo_list
#                 )
#                 sig = tb_sine_gaussian(frequency_array, A, f[i], Q[i], phi[i], dt[i], e)
#                 model["plus"]  += sig["plus"]
#                 model["cross"] += sig["cross"]
#             return model


def antenna_conversion_SNR_to_amplitude(SNR, Q, f, e, ra, dec, geocent_time, psi, ifo_list):
    """
    Function to convert SNR to amplitude of the detector response for a sine-gaussian source model.
    """
    total = 0
    for detector in ifo_list:
        coefficient = 0
        coefficient += detector.antenna_response(ra, dec, geocent_time, psi, 'plus')**2
        coefficient += e**2 * detector.antenna_response(ra,dec,geocent_time, psi, 'cross')**2
        coefficient *= Q/(2*np.sqrt(2*np.pi)*f*detector.power_spectral_density.power_spectral_density_interpolated(f))
        total += coefficient
    return SNR/np.sqrt(total)


def tb_sine_gaussian(frequency_array, amplitude, f0, Q, phi0, dt, e): # Taken from tBilby example code, TODO: update docstring
    r"""
    Our custom source model, this is just a Gaussian in frequency with
    variable global phase.

    .. math::

        \tilde{h}_{\plus}(f) = \frac{A \tau}{2\sqrt{\pi}}}
        e^{- \pi \tau (f - f_{0})^2 + i \phi_{0}} \\
        \tilde{h}_{\times}(f) = \tilde{h}_{\plus}(f) e^{i \pi / 2}


    Parameters
    ----------
    frequency_array: array-like
        The frequencies to evaluate the model at. This is required for all
        Bilby source models.
    amplitude: float
        An overall amplitude prefactor.
    f0: float
        The central frequency.
    tau: float
        The damping rate.
    phi0: float
        The reference phase.

    Returns
    -------
    dict:
        A dictionary containing "plus" and "cross" entries.

    """
    tau =  Q / (2*np.pi*f0) 
    arg = -((np.pi * tau * (frequency_array - f0)) ** 2) 
    plus = np.sqrt(np.pi) * amplitude * tau * np.exp(arg) * (np.exp(1j *phi0)+np.exp(-1j*phi0)*np.exp(-Q**2*frequency_array/f0))/ 2.0
    cross = e * plus * np.exp(1j * np.pi / 2)
    
    plus *= np.exp(-2j*frequency_array*np.pi*dt)
    cross *= np.exp(-2j*frequency_array*np.pi*dt)
    return {"plus": plus, "cross": cross}


def make_astrophysical_signal_model(n_max, ifo_list):
    """
    Factory that returns a signal_model function supporting up to n_max sine-gaussian components.

    The returned function has an explicit named-parameter signature so that Bilby can
    introspect it correctly (e.g. to build prior dicts). Internally it accepts **kwargs
    and unpacks them, so the actual call path is the same regardless of n_max.

    Parameters
    ----------
    n_max : int
        Maximum number of sine-gaussian components the model can hold.
    ifo_list : list
        Interferometer list passed through to antenna_conversion_SNR_to_amplitude.

    Returns
    -------
    callable
        A Bilby-compatible source model function.
    """
    def signal_model(frequency_array, **kwargs):
        n = int(kwargs['n'])
        SNR = [kwargs[f'SNR{i}'] for i in range(n_max)]
        f   = [kwargs[f'f{i}']   for i in range(n_max)]
        Q   = [kwargs[f'Q{i}']   for i in range(n_max)]
        phi = [kwargs[f'phi{i}'] for i in range(n_max)]
        dt  = [kwargs[f'dt{i}'] for i in range(n_max)]
        geocent_time = kwargs['geocent_time']
        psi = kwargs['psi']
        ra  = kwargs['ra']
        dec = kwargs['dec']
        e   = kwargs['e']

        model = {
            "plus":  np.zeros(frequency_array.shape, dtype='complex128'),
            "cross": np.zeros(frequency_array.shape, dtype='complex128'),
        }
        for i in range(n):
            A = antenna_conversion_SNR_to_amplitude(
                SNR[i], Q[i], f[i], e, ra, dec, geocent_time + dt[i], psi, ifo_list
            )
            sig = tb_sine_gaussian(frequency_array, A, f[i], Q[i], phi[i], dt[i], e)
            model["plus"]  += sig["plus"]
            model["cross"] += sig["cross"]
        return model

    P = inspect.Parameter
    params = [
        P('frequency_array', P.POSITIONAL_OR_KEYWORD),
        P('n',               P.POSITIONAL_OR_KEYWORD),
    ]
    for i in range(n_max):
        params.append(P(f'SNR{i}', P.POSITIONAL_OR_KEYWORD))
    for i in range(n_max):
        params.append(P(f'f{i}', P.POSITIONAL_OR_KEYWORD))
    for i in range(n_max):
        params.append(P(f'Q{i}', P.POSITIONAL_OR_KEYWORD))
    for i in range(n_max):
        params.append(P(f'phi{i}', P.POSITIONAL_OR_KEYWORD))
    for i in range(n_max):
        params.append(P(f'dt{i}', P.POSITIONAL_OR_KEYWORD))
    params += [
        P('geocent_time', P.POSITIONAL_OR_KEYWORD),
        P('psi',          P.POSITIONAL_OR_KEYWORD),
        P('ra',           P.POSITIONAL_OR_KEYWORD),
        P('dec',          P.POSITIONAL_OR_KEYWORD),
        P('e',            P.POSITIONAL_OR_KEYWORD),
        P('kwargs',       P.VAR_KEYWORD),
    ]
    signal_model.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    signal_model.__name__ = f'signal_model_{n_max}components'
    return signal_model


def _glitch_SNR_to_amplitude(SNR, Q, f0, ifo):
    """
    Convert a target optimal SNR to a sine-Gaussian amplitude for a glitch in a
    single interferometer, without any antenna-response weighting.

    Derived by setting SNR^2 = A^2 * Q / (2*sqrt(2*pi)*f0*S_n(f0)), which is the
    analytical optimal SNR of the sine-Gaussian template when it appears directly
    in the detector strain (F+ = 1, Fx = 0, no sky projection).
    """
    psd = ifo.power_spectral_density.power_spectral_density_interpolated(f0)
    return SNR / np.sqrt(Q / (2 * np.sqrt(2 * np.pi) * f0 * psd))


def make_glitch_signal_model(n_max, ifo):
    """
    Factory that returns a signal_model for glitches in a single interferometer,
    supporting up to n_max sine-gaussian components.

    No antenna response is applied: the amplitude is determined solely from the
    target SNR and the interferometer's PSD.  Create one model per detector for
    joint multi-IFO inference with independent per-IFO glitches.

    Parameters
    ----------
    n_max : int
        Maximum number of sine-gaussian components.
    ifo : bilby.gw.detector.Interferometer
        The interferometer whose PSD is used for the SNR-to-amplitude conversion.

    Returns
    -------
    callable
        A Bilby-compatible source model returning {"plus": glitch_strain, "cross": zeros}.
    """
    def signal_model(frequency_array, **kwargs):
        n = int(kwargs['n'])
        SNR = [kwargs[f'SNR{i}'] for i in range(n_max)]
        f   = [kwargs[f'f{i}']   for i in range(n_max)]
        Q   = [kwargs[f'Q{i}']   for i in range(n_max)]
        phi = [kwargs[f'phi{i}'] for i in range(n_max)]
        dt  = [kwargs[f'dt{i}']  for i in range(n_max)]

        model = {
            "plus":  np.zeros(frequency_array.shape, dtype='complex128'),
            "cross": np.zeros(frequency_array.shape, dtype='complex128'),
        }
        for i in range(n):
            A = _glitch_SNR_to_amplitude(SNR[i], Q[i], f[i], ifo)
            sig = tb_sine_gaussian(frequency_array, A, f[i], Q[i], phi[i], dt[i], e=0)
            model["plus"] += sig["plus"]
        return model

    P = inspect.Parameter
    params = [
        P('frequency_array', P.POSITIONAL_OR_KEYWORD),
        P('n',               P.POSITIONAL_OR_KEYWORD),
    ]
    for i in range(n_max):
        params.append(P(f'SNR{i}', P.POSITIONAL_OR_KEYWORD))
    for i in range(n_max):
        params.append(P(f'f{i}', P.POSITIONAL_OR_KEYWORD))
    for i in range(n_max):
        params.append(P(f'Q{i}', P.POSITIONAL_OR_KEYWORD))
    for i in range(n_max):
        params.append(P(f'phi{i}', P.POSITIONAL_OR_KEYWORD))
    for i in range(n_max):
        params.append(P(f'dt{i}', P.POSITIONAL_OR_KEYWORD))
    params.append(P('kwargs', P.VAR_KEYWORD))
    signal_model.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    signal_model.__name__ = f'glitch_model_{n_max}components'
    return signal_model
    

######################################################################################################################################
##################################################### Resonant phase shift model #####################################################
######################################################################################################################################

def bns_with_resonances_factory(n_max):
    """
    Factory for a BNS source model with up to n_max resonant phase shifts.

    Returns a bilby-compatible source model whose signature exposes all
    resonance parameters explicitly, so that WaveformGenerator can
    introspect and sample them during inference.

    The resonance parameters follow the same naming convention as
    multi_resonance_model_factory: n, f0{i}, dphi{i} for i in range(n_max).

    Parameters
    ----------
    n_max : int
        Maximum number of resonant phase-shift components.

    Returns
    -------
    callable
        A bilby frequency-domain source model returning {'plus': ..., 'cross': ...}.
    """
    resonance_model = multi_resonance_model_factory(n_max)

    def bns_with_resonances(frequency_array, **kwargs):
        m1        = kwargs['mass_1']
        m2        = kwargs['mass_2']
        a1        = kwargs['a_1']
        a2        = kwargs['a_2']
        tilt_1     = kwargs['tilt_1']
        tilt_2     = kwargs['tilt_2']
        phi_12    = kwargs['phi_12']
        phi_jl    = kwargs['phi_jl']
        lambda1   = kwargs['lambda_1']
        lambda2   = kwargs['lambda_2']
        distance  = kwargs['luminosity_distance']
        inclination = kwargs['theta_jn']
        psi       = kwargs['phase']

        resonance_kwargs = {'n': kwargs['n']}
        for i in range(n_max):
            resonance_kwargs[f'f0{i}']   = kwargs[f'f0{i}']
            resonance_kwargs[f'dphi{i}'] = kwargs[f'dphi{i}']

        resonance_phase = resonance_model(frequency_array, **resonance_kwargs)
        bns_signal = lal_binary_neutron_star(
            frequency_array, m1, m2, distance,
            a1, tilt_1, phi_12, a2, tilt_2, phi_jl,
            inclination, psi, lambda1, lambda2,
        )

        phase_factor_plus  = np.exp(1j * resonance_phase) * (0.5 * (1 + np.cos(inclination) ** 2))
        phase_factor_cross = np.exp(1j * resonance_phase) * np.cos(inclination)

        return {
            'plus':  bns_signal['plus']  * phase_factor_plus,
            'cross': bns_signal['cross'] * phase_factor_cross,
        }

    P = inspect.Parameter
    params = [
        P('frequency_array', P.POSITIONAL_OR_KEYWORD),
        P('mass_1',              P.POSITIONAL_OR_KEYWORD),
        P('mass_2',              P.POSITIONAL_OR_KEYWORD),
        P('a_1',              P.POSITIONAL_OR_KEYWORD),
        P('a_2',              P.POSITIONAL_OR_KEYWORD),
        P('tilt_1',             P.POSITIONAL_OR_KEYWORD),
        P('tilt_2',             P.POSITIONAL_OR_KEYWORD),
        P('phi_12',            P.POSITIONAL_OR_KEYWORD),
        P('phi_jl',            P.POSITIONAL_OR_KEYWORD),
        P('lambda_1',         P.POSITIONAL_OR_KEYWORD),
        P('lambda_2',         P.POSITIONAL_OR_KEYWORD),
        P('luminosity_distance',        P.POSITIONAL_OR_KEYWORD),
        P('theta_jn',     P.POSITIONAL_OR_KEYWORD),
        P('phase',             P.POSITIONAL_OR_KEYWORD),
        P('n',               P.POSITIONAL_OR_KEYWORD),
    ]
    for i in range(n_max):
        params.append(P(f'f0{i}', P.POSITIONAL_OR_KEYWORD))
    for i in range(n_max):
        params.append(P(f'dphi{i}', P.POSITIONAL_OR_KEYWORD))

    bns_with_resonances.__signature__ = inspect.Signature(params)
    bns_with_resonances.__name__ = f'bns_with_resonances_{n_max}components'
    return bns_with_resonances


def resonance_phase_shift(frequency_array, f0, delta_phi):
    """
    Phase shift defined as: 
    $\\delta\\Psi(f) = \\Theta(f-f0) (1-\\frac{f}{f0}) \\delta\\Phi$
    """

    return np.heaviside(frequency_array - f0, 0) * (1 - frequency_array/f0) * delta_phi


def multi_resonance_model_factory(n_max):
    """
    Stacks several resonant phase shifts
    """

    def signal_model(frequency_array, **kwargs):
        n = int(kwargs['n'])
        f0 = [kwargs[f'f0{i}'] for i in range(n_max)]
        dphi = [kwargs[f'dphi{i}'] for i in range(n_max)]

        model = np.zeros_like(frequency_array)
        for i in range(n):
            model += resonance_phase_shift(frequency_array, f0[i], dphi[i])
        return model

    P = inspect.Parameter
    params = [
        P('frequency_array', P.POSITIONAL_OR_KEYWORD),
        P('n',               P.POSITIONAL_OR_KEYWORD),
    ]
    for i in range(n_max):
        params.append(P(f'f0{i}', P.POSITIONAL_OR_KEYWORD))
    for i in range(n_max):
        params.append(P(f'dphi{i}', P.POSITIONAL_OR_KEYWORD))

    signal_model.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    signal_model.__name__ = f'signal_model_{n_max}components'
    return signal_model
    