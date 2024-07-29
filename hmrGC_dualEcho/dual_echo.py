import numpy as np
import json
import os

try:
    import cupy as xp
except ModuleNotFoundError:
    import numpy as xp
import warnings
from hmrGC.helper import xp_function, \
    scale_array2interval, scale_array2interval_along1D, move2cpu, move2gpu
from hmrGC.graph_cut import GraphCut
from hmrGC.image3d import Image3D, pad_array3d, depad_array3d, inpaint
from hmrGC.constants import RESCALE, GYRO


class DualEcho(Image3D):
    def __init__(self, signal, mask, params):
        Image3D.__init__(self, signal, mask, params,
                         'dixon_imaging/multi_echo.json')
        d = os.path.dirname(__file__)
        with open(os.path.join(d, "dual_echo.json"), "r") as read_file:
            self._json_file = json.load(read_file)
        self.set_options_default()
        self.set_methods_default()
        self._set_fatphasor()

        # Check input data
        assert len(self.params['TE_s']) == self.signal.shape[-1] == 2

        # Set field-map range
        TE_s = self.params['TE_s']
        dTE = np.diff(TE_s)  # FIXME: unnecessary variable definition
        period = np.abs(1 / (TE_s[1] - TE_s[0])) * 1e6 / self.params['centerFreq_Hz']

        range_fm_ppm = [-period*0.5, period*0.5]
        self.range_fm_ppm = np.array(range_fm_ppm)

    @property
    def phasormap(self):
        return self._phasormap

    @phasormap.setter
    def phasormap(self, val):
        self._phasormap = val
        if hasattr(self, 'mask'):
            self._phasormap_masked = val[self.mask]

    @property
    def fieldmap(self):
        return self._phasormap / (2 * np.pi * np.diff(self.params['TE_s'])[0])

    @property
    def phasors(self):
        return self._phasors

    @phasors.setter
    def phasors(self, val):
        self._phasors = val
        if hasattr(self, 'mask'):
            self._phasors_masked = val[self.mask]

    @property
    def phasors_masked(self):
        return self._phasors_masked

    @phasors_masked.setter
    def phasors_masked(self, val):
        self._phasors_masked = val
        if hasattr(self, 'mask'):
            phasors = np.zeros((self.mask.shape[0], self.mask.shape[1],
                                self.mask.shape[2], val.shape[-1]),
                               dtype=np.complex64) * np.nan
            phasors[self.mask] = val
            self._phasors = phasors

    @property
    def data_consistency(self):
        return self._data_consistency

    @data_consistency.setter
    def data_consistency(self, val):
        self._data_consistency = val
        if hasattr(self, 'mask'):
            self._data_consistency_masked = val[self.mask]

    @property
    def data_consistency_masked(self):
        return self._data_consistency_masked

    @data_consistency_masked.setter
    def data_consistency_masked(self, val):
        self._data_consistency_masked = val
        if hasattr(self, 'mask'):
            data_consistency = np.zeros((self.mask.shape[0], self.mask.shape[1],
                                         self.mask.shape[2], val.shape[-1]),
                                        dtype=np.float32) * np.nan
            data_consistency[self.mask] = val
            self._data_consistency = data_consistency

    @property
    def range_fm(self):
        """array, sampling range for fieldmap; *default* [-period//2, +period//2] Hz
        (uniform TE sampling) or [-600, 600] Hz (else)
        """
        return self._range_fm

    @range_fm.setter
    def range_fm(self, val):
        fm2phase = np.diff(self.params['TE_s'])[0]  # FIXME: unnecessary variable definition
        self._range_fm = val
        self._range_fm_ppm = val / self.params['centerFreq_Hz'] * 1e6

    @property
    def range_fm_ppm(self):
        """array, sampling range for fieldmap; *default* [-period//2, +period//2] Hz
        (uniform TE sampling) or [-600, 600] Hz (else)
        """
        return self._range_fm_ppm

    @range_fm_ppm.setter
    def range_fm_ppm(self, val):
        self.range_fm = val * self.params['centerFreq_Hz'] * 1e-6

    @xp_function
    def set_phasor_values(self, method='multipeak', xp=np):
        signal = move2gpu(self._signal_masked.copy(), xp).astype(xp.complex128)
        mag = xp.abs(signal) ** 2
        if method == 'multipeak':
            c = move2gpu(self.fatphasor, xp).astype(xp.complex128)
            # mag = mag.astype(np.complex64)
            w1 = xp.ones((mag.shape[0], 2))  # .astype(np.complex64)
            w2 = xp.ones((mag.shape[0], 2))  # .astype(np.complex64)
            f = xp.ones((mag.shape[0], 2))  # .astype(np.complex64)
            a1 = (c[0].real ** 2 - c[0].imag ** 2 - 2 * c[0].real * c[1].real + xp.abs(c[1]) ** 2) ** 2 + 4 * \
                 (c[0].real - c[1].real) ** 2 * c[0].imag ** 2
            a2 = 2 * (c[0].real ** 2 - c[0].imag ** 2 - 2 * c[0].real * c[1].real + xp.abs(c[1]) ** 2) * \
                 (mag[:, 0] - mag[:, 1]) - 4 * (c[0].real - c[1].real) ** 2 * mag[:, 0]
            a3 = (mag[:, 0] - mag[:, 1]) ** 2

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f[:, 0] = xp.sqrt(-a2 / (2 * a1) + xp.sqrt(a2 ** 2 / (4 * a1 ** 2) - a3 / a1))
                f[:, 1] = xp.sqrt(-a2 / (2 * a1) - xp.sqrt(a2 ** 2 / (4 * a1 ** 2) - a3 / a1))
                d0 = xp.sqrt(mag[:, 0] - c[0].imag ** 2 * f[:, 0] ** 2)
                d1 = xp.sqrt(mag[:, 0] - c[0].imag ** 2 * f[:, 1] ** 2)

            b = -c[0].real * f
            w1[:, 0] = b[:, 0] + d0
            w2[:, 0] = b[:, 0] - d0
            w1[:, 1] = b[:, 1] + d1
            w2[:, 1] = b[:, 1] - d1
            tol = xp.mean(mag) * 10 ** (-4)

            phasors_masked = xp.ones((mag.shape[0], 0))
            data_con = xp.ones((mag.shape[0], 0))
            for w in [w1, w2]:
                if self.options['neighboring_nan']:
                    w = move2gpu(self._neighboring_nan(move2cpu(w, xp)), xp)
                cond = xp.square(w) + 2 * c[1].real * w * f + xp.square(xp.abs(c[1])) * xp.square(f)
                cond[:, 0] -= mag[:, 1]
                cond[:, 1] -= mag[:, 1]
                w[xp.abs(cond) > tol] = xp.nan
                if not self.options['allow_negative_values']:
                    w[w < 0] = xp.nan
                denominator = ((w + xp.conj(c[0]) * f) * (w + c[1] * f))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tmp_phasors = xp.repeat((xp.conj(signal[:, 0]) * signal[:, 1])[:, xp.newaxis],
                                            denominator.shape[-1], axis=-1) / denominator
                    phasors_masked = xp.concatenate((phasors_masked, tmp_phasors),
                                                    axis=-1)

                tmp_data_con = w
                tmp_data_con[w > 0] = 1
                data_con = xp.concatenate((data_con, tmp_data_con), axis=-1)

        elif method == 'singlepeak':
            theta = xp.angle(move2gpu(self.fatphasor_single, xp).astype(xp.complex128))
            # mag = mag.astype(np.complex64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sqrt1 = xp.sqrt((mag[:, 0] * (xp.cos(theta[1]) - 1) -
                                 mag[:, 1] * (xp.cos(theta[0]) - 1)) /
                                (xp.cos(theta[1]) - xp.cos(theta[0])))
                sqrt2 = xp.sqrt((mag[:, 0] * (xp.cos(theta[1]) + 1) -
                                 mag[:, 1] * (xp.cos(theta[0]) + 1)) /
                                (xp.cos(theta[1]) - xp.cos(theta[0])))
            w = xp.zeros((sqrt1.shape[0], 2))
            w[:, 0] = 0.5 * (sqrt1 + sqrt2)
            w[:, 1] = 0.5 * (sqrt1 - sqrt2)
            if self.options['neighboring_nan']:
                w = move2gpu(self._neighboring_nan(move2cpu(w, xp)), xp)
            if not self.options['allow_negative_values']:
                w[w < 0] = xp.nan
            denominator = (w + w[:, ::-1] * xp.exp(-1j * theta[0])) * (w + w[:, ::-1] * xp.exp(1j * theta[1]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                phasors_masked = xp.repeat((xp.conj(signal[:, 0]) * signal[:, 1])[:, xp.newaxis],
                                           2, axis=-1) / denominator

            data_con = w
            data_con[w > 0] = 1
            data_con[w[:, ::-1] < 0] = w[:, ::-1][w[:, ::-1] < 0]

        phasors_masked = phasors_masked / xp.abs(phasors_masked)
        self.phasors_masked = move2cpu(phasors_masked, xp)
        self._phasors_original = self.phasors
        self.data_consistency_masked = move2cpu(data_con, xp)
        self._data_consistency_original = self.data_consistency

    @xp_function
    def _neighboring_nan(self, w, num_voxels=1, xp=np):
        w_image = xp.zeros((self.mask.shape[0],
                            self.mask.shape[1],
                            self.mask.shape[2],
                            w.shape[-1]), dtype=xp.float32)
        w_image[self.mask] = move2gpu(w, xp)
        meshgrid = xp.meshgrid(xp.arange(self.mask.shape[1]),  # FIXME: unnecessary variable definition
                               xp.arange(self.mask.shape[0]),
                               xp.arange(self.mask.shape[2]))

        mask = xp.zeros((self.mask.shape[0] + 2 * num_voxels,
                         self.mask.shape[1] + 2 * num_voxels,
                         self.mask.shape[2] + 2 * num_voxels,
                         w.shape[-1]))
        mask_unpadded = xp.isnan(w_image)
        for i in range(w_image.shape[-1]):
            mask[..., i] = move2gpu(
                pad_array3d(move2cpu(mask_unpadded[..., i], xp), (num_voxels, num_voxels, num_voxels)), xp)
        for i in range(num_voxels + 1):
            for j in range(num_voxels + 1):
                for k in range(num_voxels + 1):
                    padded_mask = xp.pad(mask_unpadded, ((i, 2 * num_voxels - i),
                                                         (j, 2 * num_voxels - j),
                                                         (k, 2 * num_voxels - k),
                                                         (0, 0)))
                    mask += padded_mask
                    padded_mask = xp.pad(mask_unpadded, ((i, 2 * num_voxels - i),
                                                         (j, 2 * num_voxels - j),
                                                         (2 * num_voxels - k, k),
                                                         (0, 0)))
                    mask += padded_mask
                    padded_mask = xp.pad(mask_unpadded, ((2 * num_voxels - i, i),
                                                         (2 * num_voxels - j, j),
                                                         (2 * num_voxels - k, k),
                                                         (0, 0)))
                    mask += padded_mask
                    padded_mask = xp.pad(mask_unpadded, ((2 * num_voxels - i, i),
                                                         (2 * num_voxels - j, j),
                                                         (k, 2 * num_voxels - k),
                                                         (0, 0)))
                    mask += padded_mask
                    padded_mask = xp.pad(mask_unpadded, ((i, 2 * num_voxels - i),
                                                         (2 * num_voxels - j, j),
                                                         (k, 2 * num_voxels - k),
                                                         (0, 0)))
                    mask += padded_mask
                    padded_mask = xp.pad(mask_unpadded, ((i, 2 * num_voxels - i),
                                                         (2 * num_voxels - j, j),
                                                         (2 * num_voxels - k, k),
                                                         (0, 0)))
                    mask += padded_mask
                    padded_mask = xp.pad(mask_unpadded, ((2 * num_voxels - i, i),
                                                         (j, 2 * num_voxels - j),
                                                         (2 * num_voxels - k, k),
                                                         (0, 0)))
                    mask += padded_mask
                    padded_mask = xp.pad(mask_unpadded, ((2 * num_voxels - i, i),
                                                         (j, 2 * num_voxels - j),
                                                         (k, 2 * num_voxels - k),
                                                         (0, 0)))
                    mask += padded_mask
        mask = depad_array3d(mask, (num_voxels, num_voxels, num_voxels))
        # index_nan = np.argwhere(np.isnan(w_image))
        # mask = np.zeros_like(w_image, dtype=np.uint32)
        # mask = _get_neighboring_mask(mask, meshgrid, index_nan, num_voxels)
        w_image[mask > 0] = xp.nan
        return move2cpu(w_image[self.mask], xp)

    def set_phasormap(self):
        """Calculate field-map based on mincut partition

        TODO
        """
        # Get minima arrays and convert to nodes arrays
        tmp_phasors = self.phasors_masked.copy()
        tmp_data_con = self.data_consistency_masked.copy()
        self.phasors_masked = np.angle(self.phasors_masked).astype(np.float32)

        self._unwrap_phasors()
        self._use_prior()

        nodes_arrays, nodes2pm = self._get_nodes_arrays()
        # Perform graph cut
        g = GraphCut(nodes_arrays)
        g.intra_column_scaling *= 1 / self.options['reg_param']
        g.isotropic_scaling = self.options['isotropic_weighting_factor']
        g.voxel_weighting_intra_column = self.options['noise_weighting_intra_edges']
        g.voxel_weighting_inter_column = self.options['noise_weighting_inter_edges']
        g.set_edges_and_tedges()
        g.mincut()
        self.maxflow = g.maxflow
        map_masked = g.get_map()

        phasormap_masked = nodes2pm[np.arange(len(map_masked)), map_masked]
        phasormap = np.zeros_like(self.mask, dtype=np.float32)
        phasormap[nodes_arrays['voxel_mask']] = phasormap_masked
        self.phasormap = phasormap
        self.unshift_phase()

        self.phasors_masked = tmp_phasors
        self.data_consistency_masked = tmp_data_con

    @xp_function
    def set_images(self, method="complex_approach", xp=np):
        """Calculate species images based on self.fieldmap (and self.r2starmap)

        TODO
        """
        # complex approach
        if method == "complex_approach":
            c = move2gpu(self.fatphasor, xp)
            A = xp.ones((2, 2), dtype=xp.complex64)
            A[0, 0] = c[1]
            A[0, 1] = -c[0]
            A[1, 0] = -1
            A *= 1 / (c[1] - c[0])
            x = move2gpu(self._signal_masked.copy(), xp)
            x[:, 1] *= xp.conj(xp.exp(1.0j * move2gpu(self._phasormap_masked, xp)))
            y = xp.dot(x, A.T)
            images = {}
            mask = move2gpu(self.mask, xp)
            water = xp.zeros_like(mask, dtype=xp.complex64)
            water[mask] = y[:, 0]
            images['water'] = water #np.abs(water)
            fat = xp.zeros_like(mask, dtype=xp.complex64)
            fat[mask] = y[:, 1]
            images['fat'] = fat #np.abs(fat)
        elif method == 'nonlinear_approach':
            raise NotImplementedError
        self.images = images

    def _layer_phasors(self):
        self.set_phasor_values(method=self.options['fatmodel'])

    def _layer_init(self):
       self._prior_original = [self.phasormap]
       self._prior = [self.phasormap]

    def _layer_phasormap(self):
        self.set_phasormap()

    def _layer_images(self):
        self.set_images()

    def _set_fatphasor(self):
        freqs_ppm = self.params['FatModel']['freqs_ppm']
        rel_amps = self.params['FatModel']['relAmps']
        delta_f = [0, *GYRO * freqs_ppm * self.params['fieldStrength_T']]
        c = np.zeros(2, dtype=np.complex64)
        for i, amp in enumerate(rel_amps):
            c += (amp * np.exp(2.0j * np.pi * delta_f[i + 1] * self.params['TE_s']))
        self.fatphasor = c
        freqs_ppm = np.array([freqs_ppm[np.argmax(self.params['FatModel']['relAmps'])]])
        rel_amps = np.array([1.])
        c = np.exp(2.0j * np.pi * GYRO * freqs_ppm * self.params['fieldStrength_T'] *
                   self.params['TE_s'])
        self.fatphasor_single = c

    @xp_function
    def _unwrap_phasors(self, xp=np):
        if self.options['unwrapping']:
            delta_te = np.diff(self.params['TE_s'])[0]
            unwrapping_range = 2 * np.abs(self.range_fm * delta_te)
            phasors = move2gpu(self.phasors_masked, xp)
            data_con = move2gpu(self.data_consistency_masked, xp)

            phasors_unwrapped = phasors.copy()
            data_con_unwrapped = data_con.copy()
            num_periods = (np.max(unwrapping_range) - 1) / 2
            for i in range(1, int(np.ceil(num_periods)) + 1):
                tmp_phasors = phasors - 2 * i * np.pi
                phasors_unwrapped = xp.concatenate((phasors_unwrapped, tmp_phasors),
                                                   axis=-1)
                tmp_phasors = phasors + 2 * i * np.pi
                phasors_unwrapped = xp.concatenate((phasors_unwrapped, tmp_phasors),
                                                   axis=-1)
                data_con_unwrapped = xp.concatenate((data_con_unwrapped, data_con,
                                                     data_con), axis=-1)
            phasors_unwrapped[phasors_unwrapped / (2 * np.pi * delta_te) < self.range_fm[0]] = xp.nan
            phasors_unwrapped[phasors_unwrapped / (2 * np.pi * delta_te) > self.range_fm[1]] = xp.nan
            data_con_unwrapped[phasors_unwrapped / (2 * np.pi * delta_te) < self.range_fm[0]] = xp.nan
            data_con_unwrapped[phasors_unwrapped / (2 * np.pi * delta_te) > self.range_fm[1]] = xp.nan
            self.phasors_masked = move2cpu(phasors_unwrapped, xp)
            self.data_consistency_masked = move2cpu(data_con_unwrapped, xp)

    @xp_function
    def _use_prior(self, xp=np):
        if self.options['prior'] is not False:
            ppm2fm = self.params['centerFreq_Hz'] * 1e-6
            fm2phase = np.diff(self.params['TE_s'])[0] * 2 * np.pi
            phasors = move2gpu(self.phasors_masked, xp)
            data_con = move2gpu(self.data_consistency_masked, xp)
            prior_range, prior_insert = self._get_prior()
            prior_range = move2gpu(prior_range, xp)
            neighborhood_for_range = self.options['prior']['neighborhood_for_range'] * ppm2fm * fm2phase

            unwrap_factor = (prior_range + np.pi) / (2 * np.pi)
            unwrap_factors = [xp.floor(unwrap_factor), xp.ceil(unwrap_factor)]
            tmp_phasors1 = phasors + np.pi * (2 * unwrap_factors[0][:, xp.newaxis])
            tmp_phasors2 = phasors + np.pi * (2 * unwrap_factors[1][:, xp.newaxis])
            phasors = xp.concatenate((tmp_phasors1, tmp_phasors2), axis=-1)
            data_con = xp.concatenate((data_con, data_con), axis=-1)

            prior_range = xp.repeat(prior_range[:, xp.newaxis], phasors.shape[-1],
                                    axis=-1)
            phasors[xp.abs(phasors - prior_range) > neighborhood_for_range] = xp.nan
            data_con[xp.abs(phasors - prior_range) > neighborhood_for_range] = xp.nan
            if len(self.options['prior']['layer_for_insert']) > 0:
                neighborhood_for_insert = self.options['prior']['neighborhood_for_insert'] * ppm2fm * fm2phase
                prior_insert = move2gpu(prior_insert, xp)
                for i in range(prior_insert.shape[-1]):
                    prior_insert_i = prior_insert[:, i]
                    prior_insert_i_repeated = \
                        xp.repeat(prior_insert_i[:, xp.newaxis],
                                  phasors.shape[-1], axis=-1)
                    helper_mask = \
                        xp.sum(xp.abs(phasors - prior_insert_i_repeated) < neighborhood_for_insert,
                               axis=-1) == 0
                    prior_insert_i[~helper_mask] = xp.nan
                    phasors = xp.concatenate((phasors, prior_insert_i[:, xp.newaxis]), axis=-1)
                    data_con_prior = xp.zeros_like(prior_insert_i[:, xp.newaxis])
                    data_con_prior[~helper_mask] = xp.nan
                    data_con = xp.concatenate((data_con, data_con_prior), axis=-1)
            self.phasors_masked = move2cpu(phasors, xp)
            self.data_consistency_masked = move2cpu(data_con, xp)

    @xp_function
    def _get_nodes_arrays(self, xp=np):
        phasors = move2gpu(self.phasors_masked, xp)
        data_con = move2gpu(self.data_consistency_masked, xp)
        # Calculate noise weighting
        MIP = xp.sum(xp.abs(move2gpu(self.signal[self.mask], xp)), axis=-1, dtype=xp.float32)
        noise_weighting = move2gpu(scale_array2interval(move2cpu(MIP, xp), RESCALE), xp)

        nodes2pm_argsort = xp.argsort(phasors, axis=-1)
        nodes2pm = xp.take_along_axis(phasors, nodes2pm_argsort, axis=-1)
        data_con = xp.take_along_axis(data_con, nodes2pm_argsort, axis=-1)

        # remove voxels with no phasor solution
        voxel_mask_reshaped = move2gpu(self._mask_reshaped.copy(), xp)
        helper_mask = voxel_mask_reshaped.copy()
        helper_mask[helper_mask] = \
            (xp.sum(~xp.isfinite(nodes2pm), axis=-1) == nodes2pm.shape[-1])
        voxel_mask_reshaped[helper_mask] = 0
        voxel_mask = xp.reshape(voxel_mask_reshaped, self.mask.shape)

        # change helper_mask to match shape of other nodes arrays
        helper_mask = (xp.sum(~xp.isfinite(nodes2pm), axis=-1) !=
                       nodes2pm.shape[-1])
        nodes2pm = nodes2pm[helper_mask]
        data_con = data_con[helper_mask]
        noise_weighting = noise_weighting[helper_mask]
        num_minima_per_voxel = xp.sum(xp.isfinite(nodes2pm), axis=-1)
        max_minima_per_voxel = xp.max(num_minima_per_voxel)
        nodes2pm = nodes2pm[:, :max_minima_per_voxel]
        data_con = data_con[:, :max_minima_per_voxel]

        nodes_arrays = {}
        nodes_arrays['voxel_mask'] = move2cpu(voxel_mask, xp)
        intra_column_mask = xp.isfinite(nodes2pm)
        nodes_arrays['intra_column_mask'] = move2cpu(intra_column_mask, xp)
        intra_column_cost = xp.abs(data_con)
        intra_column_cost[~intra_column_mask] = xp.nan
        intra_column_cost[num_minima_per_voxel > 1] = \
            move2gpu(scale_array2interval_along1D(move2cpu(intra_column_cost[num_minima_per_voxel > 1], xp),
                                                  RESCALE, xp), xp)
        intra_column_cost[num_minima_per_voxel == 1, 0] = 1
        nodes_arrays['intra_column_cost'] = move2cpu(intra_column_cost, xp)
        inter_column_cost = (nodes2pm - xp.nanmin(nodes2pm))
        inter_column_cost[~xp.isfinite(inter_column_cost)] = 0
        nodes_arrays['inter_column_cost'] = move2cpu(inter_column_cost * 1e2, xp)
        nodes_arrays['voxel_weighting'] = move2cpu(noise_weighting, xp)
        return nodes_arrays, move2cpu(nodes2pm, xp)

    def voxelSize2downsampling(self, voxelSize_mm):
        voxelSize_mm = np.array(voxelSize_mm)
        voxelSize_mm_original = np.array(self._voxel_size_original)
        downsampling_factor = np.floor(voxelSize_mm / voxelSize_mm_original)
        downsampling_factor[downsampling_factor < 1] = 1
        return downsampling_factor

    def inpaint_phasors(self):
        helper_mask = np.sum(np.isnan(self.phasors), axis=-1) == \
                      self.phasors.shape[-1]
        helper_mask[~self.mask] = 0
        phasors = inpaint(self.phasors, helper_mask)
        phasors[~self.mask] = np.nan
        data_con = self.data_consistency
        data_con[helper_mask] = np.nan
        helper_mask = np.sum(np.isnan(self.phasors), axis=-1) == self.phasors.shape[-1]
        self.mask[helper_mask] = False
        self.phasors = phasors
        self.data_consistency = data_con

    def _check_for_convergence(self):
        if len(self._prior) > 2 and \
                np.array_equal(self._prior[-1], self._prior[-2]):
            self._break_layer = True

    def unshift_phase(self):
        phase_shifted = self._phasormap_masked
        phase_range = np.array([np.min(phase_shifted),
                                np.max(phase_shifted)])
        phase_range /= 2 * np.pi
        self.phasormap[self.mask] = phase_shifted - 2 * np.pi * np.round(np.mean(phase_range))
