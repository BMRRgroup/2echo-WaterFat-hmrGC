import pytest
import copy
import numpy as np

from hmrGC_dualEcho.dual_echo import DualEcho

tol = 10


def test_dual_echo_init(signal, mask, params):
    params['TE_s'] = params['TE_s'][:2]
    signal = signal[:, :, :, :2]
    g = DualEcho(signal, mask, params)
    assert g._signal_masked.shape == (20 * 20 - 3, 2)
    _signal_masked = signal[mask]
    assert np.array_equal(g._signal_masked, _signal_masked)
    assert g._num_voxel == 20 * 20
    assert g._mask_reshaped.shape[0] == g._num_voxel
    np.testing.assert_almost_equal(g.fatphasor_single,
                                   np.array([-0.46431689 + 0.88566914j, 0.34288297 - 0.93937813j]))
    np.testing.assert_almost_equal(g.fatphasor,
                                   np.array([-0.38628584 + 0.6679957j, 0.3556605 - 0.539722j]))


def test_dual_echo_methods(signal, mask, params):
    params['TE_s'] = params['TE_s'][:2]
    signal = signal[:, :, :, :2]
    g = DualEcho(signal, mask, params)
    assert len(g.methods['single-res']) == 3
    assert len(g.methods['multi-res']) == 6


def test_dual_echo_set_phasor_values(signal, mask, params):
    params['TE_s'] = params['TE_s'][:2]
    signal = signal[:, :, :, :2]
    g = DualEcho(signal, mask, params)
    g.set_phasor_values(method='multipeak')
    assert g.phasors.shape[-1] == 4
    assert ~np.isfinite(g.phasors[3, 0, :, 0])
    assert ~np.isfinite(g.phasors[3, 0, :, 1])
    assert np.isfinite(g.phasors[3, 0, :, 2])
    assert np.isfinite(g.phasors[3, 0, :, 3])
    assert g.data_consistency[3, 0, 0, 3] < 0
    assert g.data_consistency[3, 0, :, 2] == 1
    assert g.data_consistency.shape == g.phasors.shape

    g = DualEcho(signal, mask, params)
    g.set_phasor_values(method='singlepeak')
    assert np.isfinite(g.phasors[3, 0, :, 0])
    assert np.isfinite(g.phasors[3, 0, :, 1])
    assert g.data_consistency[3, 0, 0, 0] < 1
    assert g.data_consistency[3, 0, 0, 1] < 1
    assert g.phasors.shape[-1] == 2
    assert g.data_consistency.shape == g.phasors.shape


def test_dual_echo_neighboring_nan(signal, mask, params):
    params['TE_s'] = params['TE_s'][:2]
    signal = signal[:, :, :, :2]
    g = DualEcho(signal, mask, params)
    g.set_phasor_values(method='singlepeak')
    w = np.zeros_like(signal)[mask]
    w[18, 0] = np.nan
    w_image = np.zeros((mask.shape[0],
                        mask.shape[1],
                        mask.shape[2],
                        w.shape[-1]), dtype=np.float32)
    w_image[mask] = w
    assert ~np.isfinite(w_image[1, 1, 0, 0])
    w_new = g._neighboring_nan(w)
    w_image[mask] = w_new
    assert ~np.isfinite(w_image[1, 0, 0, 0])
    assert ~np.isfinite(w_image[1, 2, 0, 0])
    assert ~np.isfinite(w_image[2, 0, 0, 0])
    assert ~np.isfinite(w_image[2, 1, 0, 0])
    assert ~np.isfinite(w_image[2, 2, 0, 0])
    assert np.isfinite(w_image[0, 1, 0, 0])


def test_dual_echo_set_phasormap(signal, mask, params):
    params['TE_s'] = params['TE_s'][:2]
    signal = signal[:, :, :, :2]
    g = DualEcho(signal, mask, params)
    g.set_phasor_values(method='singlepeak')
    tmp_phasors = copy.deepcopy(g.phasors)
    tmp_data_con = copy.deepcopy(g.data_consistency)
    g.set_phasormap()
    assert np.abs(g.maxflow - 1390957) <= tol
    assert g.phasormap.shape == mask.shape
    np.testing.assert_equal(g.data_consistency, tmp_data_con)
    np.testing.assert_equal(g.phasors, tmp_phasors)


def test_dual_echo_set_images(signal, mask, params):
    params['TE_s'] = params['TE_s'][:2]
    signal = signal[:, :, :, :2]
    g = DualEcho(signal, mask, params)
    g.set_phasor_values(method='multipeak')
    g.set_phasormap()
    assert not hasattr(g, 'images')
    g.set_images()
    assert 'water' in g.images.keys()
    assert 'fat' in g.images.keys()


def test_dual_echo_unwrap_phasors(signal, mask, params):
    params['TE_s'] = params['TE_s'][:2]
    signal = signal[:, :, :, :2]
    g = DualEcho(signal, mask, params)
    g.range_fm_ppm *= 2
    g.set_phasor_values(method='multipeak')
    g.phasors_masked = np.angle(g.phasors_masked)
    g.options['unwrapping'] = True
    assert g.phasors.shape[-1] == 4
    assert np.nanmax(g.phasors) < np.pi
    assert np.nanmin(g.phasors) > -np.pi
    g._unwrap_phasors()
    assert not np.nanmax(g.phasors) < np.pi
    assert not np.nanmin(g.phasors) > -np.pi
    assert np.nanmax(g.phasors) < 2 * np.pi
    assert np.nanmin(g.phasors) > -2 * np.pi
    assert g.phasors.shape[-1] == 12
    assert g.phasors.shape == g.data_consistency.shape
    assert np.sum(np.isfinite(g.data_consistency[np.isfinite(g.phasors)])) == \
           np.sum(np.isfinite(g.phasors))


def test_dual_echo_use_prior(signal, mask, params):
    params['TE_s'] = params['TE_s'][:2]
    signal = signal[:, :, :, :2]
    g = DualEcho(signal, mask, params)
    g.set_phasor_values(method='multipeak')
    g.phasors_masked = np.angle(g.phasors_masked)
    g._prior = []
    g._prior.append(2 * np.pi * np.ones_like(mask))
    g.options['prior'] = {}
    g.options['prior']['layer_for_range'] = 0
    g.options['prior']['layer_for_insert'] = []
    g.options['prior']['neighborhood_for_range'] = 1
    assert np.nanmax(g.phasors) < np.pi
    assert np.nanmin(g.phasors) > -np.pi
    g._use_prior()
    assert np.nanmax(g.phasors) < 3 * np.pi
    assert np.nanmin(g.phasors) > -np.pi

    g = DualEcho(signal, mask, params)
    g.set_phasor_values(method='multipeak')
    g.phasors_masked = np.angle(g.phasors_masked)
    g._prior = []
    g._prior.append(2 * np.pi * np.ones_like(mask))
    assert np.sum(np.sum(np.isfinite(g.phasors), axis=-1)[mask] == 0) > 0
    g.options['prior'] = {}
    g.options['prior']['layer_for_range'] = 0
    g.options['prior']['layer_for_insert'] = [0]
    g.options['prior']['neighborhood_for_range'] = 1
    g.options['prior']['neighborhood_for_insert'] = 1
    g._use_prior()
    assert np.sum(np.sum(np.isfinite(g.phasors), axis=-1)[mask] == 0) == 0


def test_dual_echo_get_nodes_arrays(signal, mask, params):
    params['TE_s'] = params['TE_s'][:2]
    signal = signal[:, :, :, :2]
    g = DualEcho(signal, mask, params)
    g.set_phasor_values(method='multipeak')
    g.phasors_masked = np.angle(g.phasors_masked)
    nodes_arrays, nodes2pm = g._get_nodes_arrays()
    assert 'voxel_mask' in nodes_arrays.keys()
    assert 'intra_column_mask' in nodes_arrays.keys()
    assert 'intra_column_cost' in nodes_arrays.keys()
    assert 'inter_column_cost' in nodes_arrays.keys()
    assert 'voxel_weighting' in nodes_arrays.keys()
    intra_column_mask = nodes_arrays['intra_column_mask']
    assert np.nansum(nodes_arrays['intra_column_cost'][~intra_column_mask]) == 0
    assert np.nansum(nodes_arrays['inter_column_cost'][~intra_column_mask]) == 0


def test_dual_echo_voxelSize2downsampling(signal, mask, params):
    params['TE_s'] = params['TE_s'][:2]
    signal = signal[:, :, :, :2]
    g = DualEcho(signal, mask, params)
    factor = g.voxelSize2downsampling(3)
    np.testing.assert_array_equal(factor, 2)
    factor = g.voxelSize2downsampling(1.1)
    np.testing.assert_array_equal(factor, 1)


def test_dual_echo_inpaint_phasors(signal, mask, params):
    params['TE_s'] = params['TE_s'][:2]
    signal = signal[:, :, :, :2]
    g = DualEcho(signal, mask, params)
    g.set_phasor_values(method='multipeak')
    num_nan_before = np.count_nonzero(np.isnan(g.phasors))
    g.inpaint_phasors()
    num_nan_after = np.count_nonzero(np.isnan(g.phasors))
    assert num_nan_before > num_nan_after


def test_dual_echo_perform(signal, mask, params, fieldmap,
                           test_gpu):
    if test_gpu:
        params['TE_s'] = params['TE_s'][:2]
        signal = signal[:, :, :, :2]
        g = DualEcho(signal, mask, params)
        g.perform('single-res')
        assert np.abs(g.maxflow - 1390957) <= tol

        g = DualEcho(signal, mask, params)
        g.perform('multi-res')
        assert np.abs(g.maxflow - 752279) <= tol
        assert np.mean(np.abs(g.fieldmap - fieldmap)[mask]) < 15
