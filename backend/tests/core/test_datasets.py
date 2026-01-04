from backend.core.datasets import (
    generate_translations,
    make_and_dataset_pm1,
    make_or_dataset_pm1,
    make_shape_dataset,
    make_tinygps_dataset_1d_from_coords,
    make_tinygps_dataset_1d_multi,
    make_tinygps_dataset_2d,
    make_xor_dataset_pm1,
    tinygps_city_coords,
    tinygps_city_coords_exercise,
)


def _labels(samples):
    return [s["y"] for s in samples]


def test_or_and_xor_labels():
    assert _labels(make_or_dataset_pm1()) == [1, 1, 1, -1]
    assert _labels(make_and_dataset_pm1()) == [-1, -1, 1, -1]
    assert _labels(make_xor_dataset_pm1()) == [1, 1, -1, -1]


def test_generate_translations_counts():
    shape = [[1]]
    positions = generate_translations(shape, board_h=2, board_w=2)
    assert len(positions) == 4
    for grid, pos in positions:
        assert len(grid) == 2
        assert len(grid[0]) == 2
        assert pos in [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_make_shape_dataset_translation_samples():
    good = [[1]]
    bad = [[-1]]
    samples = make_shape_dataset(good, bad, board_size=(2, 2), translations=True)
    # 4 translations for each of good and bad
    assert len(samples) == 8
    for s in samples:
        assert len(s["x"]) == 4
        assert set(s["x"]).issubset({-1, 1})
        assert s["y"] in (-1, 1)
        assert "grid" in s and "pos" in s


def test_tinygps_city_coords():
    coords = tinygps_city_coords()
    assert "paris" in coords and "madrid" in coords
    assert len(coords["paris"]) == 5
    exercise = tinygps_city_coords_exercise()
    assert len(exercise["paris"]) == 2


def test_tinygps_dataset_1d_multi():
    samples = make_tinygps_dataset_1d_multi(["madrid", "paris", "berlin"])
    assert len(samples) == 15
    assert all(len(s["x"]) == 1 for s in samples)
    assert set(s["y"] for s in samples) == {0, 1, 2}


def test_tinygps_dataset_2d():
    samples = make_tinygps_dataset_2d(["paris", "madrid", "berlin", "barcelona"])
    assert len(samples) == 20
    assert all(len(s["x"]) == 2 for s in samples)
    assert set(s["y"] for s in samples) == {0, 1, 2, 3}
    first = samples[0]
    assert first["x"][0] == first["lat"]
    assert first["x"][1] == first["lon"]


def test_tinygps_dataset_1d_exercise_coords():
    coords = tinygps_city_coords_exercise()
    samples = make_tinygps_dataset_1d_from_coords(coords, ["paris", "madrid"])
    assert len(samples) == 4
