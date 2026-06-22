import numpy as np

from .base import GravitationalWaveTransient
from ...core.likelihood import _fallback_to_parameters


class GlitchGravitationalWaveTransient(GravitationalWaveTransient):
    """
    Extends GravitationalWaveTransient to support independent per-IFO
    sine-Gaussian glitch models on top of a shared GW signal.

    For each IFO that has an entry in glitch_waveform_generators the total
    template is h_total = h_gw + h_glitch, so the log-likelihood picks up
    the extra terms:

        <d | h_glitch>  and  <h_gw | h_glitch>  (cross term)

    The glitch strain is taken directly from the waveform generator output
    without any antenna-response projection, which is correct for
    detector-intrinsic artefacts.

    Parameters
    ----------
    interferometers : list or InterferometerList
    waveform_generator : WaveformGenerator
        Shared GW signal generator.
    glitch_waveform_generators : dict[str, WaveformGenerator], optional
        Maps IFO name (e.g. 'H1') to a WaveformGenerator backed by
        make_glitch_signal_model(..., ifo).  Omit an IFO to treat it as
        glitch-free.
    **kwargs
        Forwarded to GravitationalWaveTransient.__init__.
        Marginalisations (time / distance / phase / calibration) raise
        ValueError when glitch generators are provided.

    Notes
    -----
    Glitch parameters for IFO 'H1' must appear in the sampler's flat
    parameter dict with the prefix 'H1_', e.g. H1_n, H1_SNR0, H1_f0.
    _glitch_parameters() converts these to the nested dict
    {'H1': {'n': ..., 'SNR0': ..., ...}, ...} before forwarding to each
    IFO's waveform generator.
    """

    def __init__(self, interferometers, waveform_generator,
                 glitch_waveform_generators=None, **kwargs):
        if glitch_waveform_generators and any([
            kwargs.get('time_marginalization'),
            kwargs.get('distance_marginalization'),
            kwargs.get('phase_marginalization'),
            kwargs.get('calibration_marginalization'),
        ]):
            raise ValueError(
                "Marginalisations are not supported alongside per-IFO glitch models.")
        super().__init__(interferometers, waveform_generator, **kwargs)
        self.glitch_waveform_generators = glitch_waveform_generators or {}
        for gen in self.glitch_waveform_generators.values():
            for attr in ('duration', 'sampling_frequency', 'start_time'):
                setattr(gen, attr, getattr(self.interferometers, attr))

    def _glitch_parameters(self, parameters):
        """
        Reorganise the flat sampler parameter dict into a nested structure.

        Returns
        -------
        dict[str, dict]
            {ifo_name: {unprefixed_param: value, ...}} for every IFO in
            glitch_waveform_generators.
        """
        result = {}
        for ifo_name in self.glitch_waveform_generators:
            prefix = ifo_name + '_'
            result[ifo_name] = {
                k[len(prefix):]: v
                for k, v in parameters.items()
                if k.startswith(prefix)
            }
        return result

    def calculate_snrs(self, waveform_polarizations, interferometer,
                       return_array=True, parameters=None):
        parameters = _fallback_to_parameters(self, parameters)

        # Project GW onto this detector once; reuse for both GW terms and
        # the <h_gw|h_glitch> cross term rather than calling
        # _compute_full_waveform a second time via super().calculate_snrs().
        gw_strain = self._compute_full_waveform(
            waveform_polarizations, interferometer, parameters)

        d_inner_h = interferometer.inner_product(signal=gw_strain)
        h_h = float(interferometer.optimal_snr_squared(signal=gw_strain).real)

        ifo_name = interferometer.name
        if ifo_name in self.glitch_waveform_generators:
            glitch_params = self._glitch_parameters(parameters)[ifo_name]
            glitch_pols = self.glitch_waveform_generators[ifo_name] \
                .frequency_domain_strain(glitch_params)
            if glitch_pols is not None:
                glitch_strain = glitch_pols['plus']
                d_inner_h += interferometer.inner_product(signal=glitch_strain)
                h_h += float(
                    interferometer.optimal_snr_squared(signal=glitch_strain).real)
                h_h += 2 * float(
                    interferometer.template_template_inner_product(
                        gw_strain, glitch_strain).real)

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h,
            optimal_snr_squared=h_h,
            complex_matched_filter_snr=d_inner_h / h_h ** 0.5 if h_h > 0 else 0j,
        )
