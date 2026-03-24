"""Tests for dataset utilities."""

import pytest
import numpy as np

from data import extract_subject_id, Augmenter
from data.sadt import rt_to_drowsiness_index, extract_subject_id_sadt
from data.mpddf import map_fatigue_levels, extract_subject_id_mpddf


def test_extract_subject_id_seedvig():
    assert extract_subject_id("1_20151124_session.mat") == 1
    assert extract_subject_id("23_20160301_run.mat") == 23
    assert extract_subject_id("no_match.mat") == -1


def test_extract_subject_id_sadt():
    assert extract_subject_id_sadt("sub01_sess01.mat") == 1
    assert extract_subject_id_sadt("sub27_sess02.mat") == 27
    assert extract_subject_id_sadt("subject_5_session_1.mat") == 5


def test_extract_subject_id_mpddf():
    assert extract_subject_id_mpddf("sub01.mat") == 1
    assert extract_subject_id_mpddf("sub50.mat") == 50


def test_rt_to_drowsiness_index():
    # Min RT -> DI near 0
    di = rt_to_drowsiness_index(np.array([0.3]))
    assert abs(di[0]) < 0.05

    # Max RT -> DI near 1
    di = rt_to_drowsiness_index(np.array([3.0]))
    assert abs(di[0] - 1.0) < 0.05

    # Monotonic
    rts = np.array([0.3, 0.5, 1.0, 2.0, 3.0])
    dis = rt_to_drowsiness_index(rts)
    assert all(dis[i] <= dis[i + 1] for i in range(len(dis) - 1))


def test_map_fatigue_levels():
    levels = np.array([1, 2, 3, 4, 5])
    continuous = map_fatigue_levels(levels, n_levels=5)

    assert abs(continuous[0] - 0.0) < 1e-6
    assert abs(continuous[-1] - 1.0) < 1e-6
    assert all(continuous[i] <= continuous[i + 1] for i in range(len(continuous) - 1))


def test_augmenter_preserves_shape():
    import torch
    aug = Augmenter(noise_std=0.05, band_drop_prob=0.5)
    x = torch.randn(5, 17, 100)  # (B, C, T)
    x_aug = aug(x.clone())
    assert x_aug.shape == x.shape
