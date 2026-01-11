"""Datasets for perceptron demos."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

Sample = dict


def _pm1(val: int) -> int:
    return 1 if val > 0 else -1


def _flatten(grid: Sequence[Sequence[int]]) -> list[int]:
    return [cell for row in grid for cell in row]


def make_or_dataset_pm1() -> list[Sample]:
    # Inputs are in {-1, +1}. OR is positive if any input is +1.
    data = [
        ([-1, 1], 1),
        ([1, -1], 1),
        ([1, 1], 1),
        ([-1, -1], -1),
    ]
    return [{"x": x, "y": y} for x, y in data]


def make_and_dataset_pm1() -> list[Sample]:
    # AND is positive only if both inputs are +1.
    data = [
        ([-1, 1], -1),
        ([1, -1], -1),
        ([1, 1], 1),
        ([-1, -1], -1),
    ]
    return [{"x": x, "y": y} for x, y in data]


def make_xor_dataset_pm1() -> list[Sample]:
    # XOR is positive if exactly one input is +1.
    data = [
        ([-1, 1], 1),
        ([1, -1], 1),
        ([1, 1], -1),
        ([-1, -1], -1),
    ]
    return [{"x": x, "y": y} for x, y in data]


def generate_translations(
    shape_mask: Sequence[Sequence[int]],
    board_h: int,
    board_w: int,
) -> list[tuple[list[list[int]], tuple[int, int]]]:
    """Return all valid translations of a shape on a board.

    Cells belonging to the shape are +1, empty cells are -1.
    Returns list of (grid, (top, left)).
    """
    shape_h = len(shape_mask)
    shape_w = len(shape_mask[0]) if shape_h > 0 else 0
    if shape_h == 0 or shape_w == 0:
        raise ValueError("shape_mask must be non-empty")
    if board_h < shape_h or board_w < shape_w:
        raise ValueError("board must be at least as large as shape")

    positions: list[tuple[list[list[int]], tuple[int, int]]] = []
    for top in range(board_h - shape_h + 1):
        for left in range(board_w - shape_w + 1):
            grid = [[-1 for _ in range(board_w)] for _ in range(board_h)]
            for r in range(shape_h):
                for c in range(shape_w):
                    if _pm1(shape_mask[r][c]) == 1:
                        grid[top + r][left + c] = 1
            positions.append((grid, (top, left)))
    return positions


def make_shape_dataset(
    good_mask: Sequence[Sequence[int]],
    bad_mask: Sequence[Sequence[int]],
    board_size: tuple[int, int],
    translations: bool = True,
) -> list[Sample]:
    """Create a dataset of good vs bad shapes on a grid.

    Each sample includes:
    - x: flattened grid in {-1, +1}
    - y: label (+1 for good, -1 for bad)
    - grid: 2D grid for visualization
    - pos: (top, left) if translations are used
    """
    board_h, board_w = board_size
    samples: list[Sample] = []

    def _add(mask: Sequence[Sequence[int]], label: int) -> None:
        if translations:
            for grid, pos in generate_translations(mask, board_h, board_w):
                samples.append({"x": _flatten(grid), "y": label, "grid": grid, "pos": pos})
        else:
            if len(mask) != board_h or len(mask[0]) != board_w:
                raise ValueError("mask must match board size when translations=False")
            grid = [[_pm1(cell) for cell in row] for row in mask]
            samples.append({"x": _flatten(grid), "y": label, "grid": grid, "pos": (0, 0)})

    _add(good_mask, 1)
    _add(bad_mask, -1)
    return samples


CITY_COORDS = {
    "madrid": [
        (40.4167, -3.7033),  # Center of Madrid
        (40.4153, -3.6835),  # Retiro Park
        (40.4180, -3.7143),  # Royal Palace
        (40.4138, -3.6921),  # Prado Museum
        (40.4169, -3.7033),  # Puerta del Sol
    ],
    "paris": [
        (48.8575, 2.3514),  # Center of Paris
        (48.8584, 2.2945),  # Eiffel Tower
        (48.8530, 2.3499),  # Notre Dame
        (48.8606, 2.3376),  # Louvre
        (48.8606, 2.3522),  # Centre Pompidou
    ],
    "berlin": [
        (52.5200, 13.4050),  # Center of Berlin
        (52.5163, 13.3777),  # Brandenburg Gate
        (52.5169, 13.4019),  # Museum Island
        (52.5074, 13.3904),  # Checkpoint Charlie
        (52.5251, 13.3694),  # Berlin Central Station
    ],
    "barcelona": [
        (41.3874, 2.1686),  # Center of Barcelona
        (41.4036, 2.1744),  # Sagrada Familia
        (41.3819, 2.1773),  # Gothic Quarter
        (41.4145, 2.1527),  # Park Guell
        (41.3809, 2.1228),  # Camp Nou
    ],
}

CITY_COORDS_EXERCISE = {
    "paris": [
        (48.8575, 2.3514),
        (48.8584, 2.2945),
    ],
    "madrid": [
        (40.4167, -3.7033),
        (40.4153, -3.6835),
    ],
    "berlin": [
        (52.5200, 13.4050),
        (52.5163, 13.3777),
    ],
}


def tinygps_city_coords() -> dict[str, list[tuple[float, float]]]:
    """Return the TinyGPS city coordinate fixtures as (lat, lon)."""
    return CITY_COORDS


def tinygps_city_coords_exercise() -> dict[str, list[tuple[float, float]]]:
    """Return TinyGPS exercise-only coordinates (2 samples per city)."""
    return CITY_COORDS_EXERCISE


def make_tinygps_dataset_1d(city_a: str, city_b: str) -> list[Sample]:
    """Longitude-only TinyGPS dataset for two cities."""
    return make_tinygps_dataset_1d_multi([city_a, city_b])


def make_tinygps_dataset_1d_multi(cities: Iterable[str]) -> list[Sample]:
    """Longitude-only TinyGPS dataset for multiple cities.

    Returns samples with x=[lon] and y as the city index order.
    """
    city_list = list(cities)
    if not city_list:
        raise ValueError("cities must be non-empty")
    for city in city_list:
        if city not in CITY_COORDS:
            raise ValueError("unknown city name for TinyGPS dataset")
    samples: list[Sample] = []
    for label, city in enumerate(city_list):
        for lat, lon in CITY_COORDS[city]:
            samples.append({"x": [lon], "y": label, "city": city, "lat": lat, "lon": lon})
    return samples


def make_tinygps_dataset_1d_from_coords(
    coords: dict[str, list[tuple[float, float]]], cities: Iterable[str]
) -> list[Sample]:
    city_list = list(cities)
    if not city_list:
        raise ValueError("cities must be non-empty")
    for city in city_list:
        if city not in coords:
            raise ValueError("unknown city name for TinyGPS dataset")
    samples: list[Sample] = []
    for label, city in enumerate(city_list):
        for lat, lon in coords[city]:
            samples.append({"x": [lon], "y": label, "city": city, "lat": lat, "lon": lon})
    return samples


def make_baarle_hertog_dataset(
    n_samples: int = 1000,
    seed: int = 42,
    image_path: str | None = None,
) -> list[Sample]:
    """Generate 2D classification dataset from Baarle-Hertog map.

    Yellow regions = class 1 (Hertog/Belgium)
    Cream/background = class 0 (Nassau/Netherlands)

    If the image cannot be loaded, falls back to pre-computed samples.
    Returns samples with x=[x1, x2] normalized to [-1, 1].
    """
    import random

    rng = random.Random(seed)

    # Try to load image with PIL (more common than cv2)
    samples: list[Sample] = []
    loaded_from_image = False

    if image_path:
        try:
            from PIL import Image

            img = Image.open(image_path).convert("RGB")
            width, height = img.size
            pixels = img.load()

            # Collect yellow and cream pixels
            yellow_points: list[tuple[int, int]] = []
            cream_points: list[tuple[int, int]] = []

            for y in range(height):
                for x in range(width):
                    pixel = pixels[x, y]  # type: ignore[index]
                    # Pixel can be int (grayscale) or tuple (RGB/RGBA)
                    if isinstance(pixel, (tuple, list)):
                        r, g, b = pixel[0], pixel[1], pixel[2]
                    else:
                        r = g = b = int(pixel)  # Grayscale
                    # Yellow: high R, high G, low B (the enclave regions)
                    if r > 200 and g > 200 and b < 100:
                        yellow_points.append((x, y))
                    # Cream/beige background: high R, high-ish G, medium B
                    elif r > 230 and g > 220 and b > 180 and b < 240:
                        cream_points.append((x, y))

            if yellow_points and cream_points:
                n_per_class = n_samples // 2

                # Sample from each class
                yellow_sample = rng.sample(yellow_points, min(n_per_class, len(yellow_points)))
                cream_sample = rng.sample(cream_points, min(n_per_class, len(cream_points)))

                # Normalize to [-1, 1]
                for px, py in yellow_sample:
                    x1 = (px / width) * 2 - 1
                    x2 = (py / height) * 2 - 1
                    x2 = -x2  # Flip Y axis (image coords are top-down)
                    samples.append({"x": [x1, x2], "y": 1})

                for px, py in cream_sample:
                    x1 = (px / width) * 2 - 1
                    x2 = (py / height) * 2 - 1
                    x2 = -x2
                    samples.append({"x": [x1, x2], "y": 0})

                loaded_from_image = True
        except Exception:
            pass  # Fall back to synthetic data

    if not loaded_from_image:
        # Fallback: Generate synthetic "enclave-like" data
        # Create a complex pattern with multiple disconnected regions
        n_per_class = n_samples // 2

        # Class 1 (Hertog): Multiple circular/rectangular regions
        enclave_centers = [
            (-0.5, -0.4, 0.3),   # (x, y, radius) - main region
            (0.3, 0.2, 0.25),    # upper right
            (-0.2, 0.5, 0.15),   # small upper
            (0.5, -0.3, 0.18),   # lower right
            (-0.6, 0.3, 0.12),   # small left
            (0.1, -0.6, 0.2),    # lower middle
            (0.6, 0.5, 0.1),     # tiny upper right
            (-0.3, -0.7, 0.15),  # lower left
        ]

        for _ in range(n_per_class):
            # Pick a random enclave
            cx, cy, radius = rng.choice(enclave_centers)
            # Sample within the enclave (uniform in circle)
            angle = rng.uniform(0, 2 * 3.14159)
            r = radius * (rng.random() ** 0.5)  # sqrt for uniform area
            x1 = cx + r * (angle / abs(angle + 0.001)) * rng.uniform(-1, 1) * 0.7
            x2 = cy + r * rng.uniform(-1, 1) * 0.7
            x1 = max(-1, min(1, cx + rng.uniform(-radius, radius)))
            x2 = max(-1, min(1, cy + rng.uniform(-radius, radius)))
            samples.append({"x": [x1, x2], "y": 1})

        # Class 0 (Nassau): Points not in enclaves
        attempts = 0
        count = 0
        while count < n_per_class and attempts < n_per_class * 10:
            x1 = rng.uniform(-1, 1)
            x2 = rng.uniform(-1, 1)
            # Check if point is outside all enclaves
            in_enclave = False
            for cx, cy, radius in enclave_centers:
                if (x1 - cx) ** 2 + (x2 - cy) ** 2 < radius ** 2:
                    in_enclave = True
                    break
            if not in_enclave:
                samples.append({"x": [x1, x2], "y": 0})
                count += 1
            attempts += 1

    rng.shuffle(samples)
    return samples


def make_tinygps_dataset_2d(cities: Iterable[str]) -> list[Sample]:
    """Latitude+longitude TinyGPS dataset for multiple cities.

    Returns samples with x=[x1, x2] where x1=lon and x2=lat, normalized
    by subtracting the center of Paris (as per the book).
    y is the city index order.
    """
    city_list = list(cities)
    if not city_list:
        raise ValueError("cities must be non-empty")
    for city in city_list:
        if city not in CITY_COORDS:
            raise ValueError("unknown city name for TinyGPS dataset")

    # Paris center for normalization (per the book)
    paris_center_lat, paris_center_lon = 48.8575, 2.3514

    samples: list[Sample] = []
    for label, city in enumerate(city_list):
        for lat, lon in CITY_COORDS[city]:
            # x1 = normalized longitude, x2 = normalized latitude (book convention)
            x1 = lon - paris_center_lon
            x2 = lat - paris_center_lat
            samples.append({
                "x": [x1, x2],
                "y": label,
                "city": city,
                "lat": lat,
                "lon": lon,
            })
    return samples
