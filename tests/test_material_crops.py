import numpy as np
import pytest

from rgb.material_crops import material_crop


def test_interior_uses_foreground_and_moves_context_bounds():
    image = np.zeros((80, 100, 3), np.uint8)
    mask = np.zeros((80, 100), bool)
    mask[20:60, 60:90] = True
    image[mask] = (30, 150, 240)
    pixels, bounds = material_crop(image, [0, 0, 100, 80], 'interior', mask, size=32)
    assert bounds[0] >= 60 and bounds[1] >= 20
    assert bounds[2] <= 90 and bounds[3] <= 60
    assert np.all(pixels == (30, 150, 240))


def test_small_surface_fails_instead_of_sampling_background():
    image = np.zeros((80, 100, 3), np.uint8)
    mask = np.zeros((80, 100), bool)
    mask[10:20, 10:20] = True
    with pytest.raises(ValueError, match='Insufficient'):
        material_crop(image, [0, 0, 100, 80], 'interior', mask)


def test_foreground_letterbox_suppresses_background_and_preserves_shape():
    image = np.full((20, 40, 3), 240, np.uint8)
    mask = np.zeros((20, 40), bool)
    mask[:, :20] = True
    pixels, bounds = material_crop(image, [0, 0, 40, 20], 'foreground', mask, size=40)
    assert bounds == [0, 0, 40, 20]
    assert np.all(pixels[:10] == 127)
    assert np.all(pixels[10:30, 20:] == 127)
    assert np.all(pixels[10:30, :20] == 240)


def test_source_mask_must_match_frame():
    image = np.zeros((20, 40, 3), np.uint8)
    with pytest.raises(ValueError, match='source-aligned'):
        material_crop(image, [0, 0, 40, 20], 'foreground', np.ones((20, 20), bool))
