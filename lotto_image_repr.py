"""Phase 7: reversible 6-of-49 binary image and spectral representations.

The pixel layout is a reversible encoding, not a claim that lottery draws
have physical two-dimensional geometry. Each observed main-six draw maps
to exactly six active pixels. The bonus number is never included.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.fft import dctn, idctn

N, K, SIDE = 49, 6, 7
MAPPINGS = ("sequential", "snake", "permuted")
DOMAINS = ("spatial", "dct", "fft")
VERSION = "0.1.0"


def cell_to_number(mapping: str = "sequential", *, seed: int = 649) -> np.ndarray:
    """Return a 7x7 bijection from raster cell positions to numbers 1..49."""
    if mapping not in MAPPINGS or not isinstance(seed, int) or seed < 0:
        raise ValueError("invalid mapping name or permutation seed")
    result = np.arange(1, N + 1, dtype=np.int64).reshape(SIDE, SIDE)
    if mapping == "snake":
        result[1::2] = result[1::2, ::-1]
    elif mapping == "permuted":
        result = np.random.default_rng(seed).permutation(result.ravel()).reshape(SIDE, SIDE)
    if not np.array_equal(np.sort(result.ravel()), np.arange(1, N + 1)):
        raise AssertionError("mapping must be a bijection")
    return result


def validate_draws(draws: np.ndarray) -> tuple[np.ndarray, bool]:
    """Validate distinct six-number main draws without sorting the input."""
    arr = np.asarray(draws)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != K or arr.shape[0] < 1:
        raise ValueError("expected [6] or [draws,6] array")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("main draw numbers must be integers")
    values = arr.astype(np.int64, copy=False)
    if (np.any(values < 1) or np.any(values > N)
            or np.any(np.diff(np.sort(values, axis=1), axis=1) == 0)):
        raise ValueError("each main draw must contain six different numbers in 1..49")
    return values, single


def encode_draws(draws: np.ndarray, *, mapping: str = "sequential",
                 seed: int = 649) -> np.ndarray:
    """Return uint8 raster images [7,7] or [draws,7,7], exactly six ones."""
    values, single = validate_draws(draws)
    layout = cell_to_number(mapping, seed=seed).ravel()
    inverse = np.argsort(layout)
    image = np.zeros((len(values), N), dtype=np.uint8)
    image[np.arange(len(values))[:, None], inverse[values - 1]] = 1
    result = image.reshape((-1, SIDE, SIDE))
    return result[0] if single else result


def decode_images(images: np.ndarray, *, mapping: str = "sequential",
                  seed: int = 649) -> np.ndarray:
    """Invert binary masks; any non-binary, non-six-point image is rejected."""
    values = np.asarray(images)
    single = values.ndim == 2
    if single:
        values = values[None]
    if values.ndim != 3 or values.shape[1:] != (SIDE, SIDE) or not len(values):
        raise ValueError("expected [7,7] or [draws,7,7] images")
    if not np.issubdtype(values.dtype, np.number) or not np.isrealobj(values):
        raise ValueError("images must be real numeric binary masks")
    if (not np.all(np.isfinite(values))
            or not np.all((values == 0) | (values == 1))
            or not np.all(values.sum(axis=(1, 2)) == K)):
        raise ValueError("every image must contain exactly six binary active pixels")
    layout = cell_to_number(mapping, seed=seed).ravel()
    result = np.stack([np.sort(layout[row.ravel() == 1]) for row in values])
    return result[0] if single else result


def transform_images(images: np.ndarray, *, domain: str) -> np.ndarray:
    """Full, untruncated orthonormal transform of one or more 7x7 masks.

    DCT yields the full real 7x7 coefficients; FFT returns separate real and
    imaginary planes (...,2,7,7). Nothing is cropped, so both are reversible.
    """
    x = np.asarray(images, dtype=np.float64)
    if x.ndim < 2 or x.shape[-2:] != (SIDE, SIDE) or not np.all(np.isfinite(x)):
        raise ValueError("expected finite images ending in 7x7")
    if domain == "spatial":
        return x.copy()
    if domain == "dct":
        return dctn(x, type=2, norm="ortho", axes=(-2, -1))
    if domain == "fft":
        spectrum = np.fft.fft2(x, axes=(-2, -1), norm="ortho")
        return np.stack((spectrum.real, spectrum.imag), axis=-3)
    raise ValueError("unknown transform domain")


def inverse_transform(coefficients: np.ndarray, *, domain: str) -> np.ndarray:
    """Reconstruct full floating point spatial masks from lossless coefficients."""
    x = np.asarray(coefficients, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("coefficients must be finite")
    if domain in ("spatial", "dct"):
        if x.ndim < 2 or x.shape[-2:] != (SIDE, SIDE):
            raise ValueError("expected coefficients ending in 7x7")
        return x.copy() if domain == "spatial" else idctn(
            x, type=2, norm="ortho", axes=(-2, -1))
    if domain == "fft":
        if x.ndim < 3 or x.shape[-3:] != (2, SIDE, SIDE):
            raise ValueError("expected FFT coefficients ending in 2x7x7")
        reconstructed = np.fft.ifft2(x[..., 0, :, :] + 1j * x[..., 1, :, :],
                                     axes=(-2, -1), norm="ortho")
        if np.max(np.abs(reconstructed.imag)) > 1e-9:
            raise ValueError("FFT coefficients are not conjugate symmetric")
        return reconstructed.real
    raise ValueError("unknown transform domain")


def temporal_windows(draws: np.ndarray, *, window: int = 5,
                     mapping: str = "sequential", domain: str = "spatial",
                     seed: int = 649, as_volume: bool = False
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For target t only use draws [t-window,t); never put t into features.

    Standard output: [samples, window*(2 if FFT else 1),7,7] channels.
    Volume output: [samples,1,window,7,7] for temporal 3D convolution.
    Targets are canonical sorted six-number sets, independent of pixel layout.
    """
    numbers, _ = validate_draws(draws)
    if not isinstance(window, int) or not 1 <= window < len(numbers):
        raise ValueError("window must be 1..draws-1")
    if domain not in DOMAINS or (as_volume and domain != "spatial"):
        raise ValueError("volume mode requires the spatial domain")
    images = encode_draws(numbers, mapping=mapping, seed=seed)
    origins = np.arange(window, len(numbers), dtype=np.int64)
    stack = np.stack([images[t - window:t] for t in origins]).astype(np.float64)
    if as_volume:
        features = stack[:, None]
    elif domain == "fft":
        transformed = transform_images(stack, domain="fft")
        features = transformed.reshape(len(origins), window * 2, SIDE, SIDE)
    else:
        features = transform_images(stack, domain=domain)
    return (features.astype(np.float32),
            np.sort(numbers[origins].copy(), axis=1), origins)


def ascii_image(numbers: np.ndarray, *, mapping: str = "sequential") -> str:
    """Print a canonical seven-row binary view for visual inspection."""
    mask = encode_draws(numbers, mapping=mapping)
    if mask.ndim != 2:
        raise ValueError("ASCII view requires one six-number draw")
    return "\n".join(" ".join("#" if v else "." for v in row) for row in mask)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 reversible 7x7 image self-test")
    parser.add_argument("--numbers", nargs=6, type=int,
                        default=[5, 9, 26, 32, 41, 48])
    parser.add_argument("--mapping", choices=MAPPINGS, default="sequential")
    parser.add_argument("--seed", type=int, default=649)
    args = parser.parse_args()
    original = np.asarray(args.numbers, dtype=np.int64)
    mask = encode_draws(original, mapping=args.mapping, seed=args.seed)
    decoded = decode_images(mask, mapping=args.mapping, seed=args.seed)
    if not np.array_equal(decoded, np.sort(original)):
        raise AssertionError("image representation round trip failed")
    print(f"Lotto649 Image Representation v{VERSION}")
    print(f"[MAPPING] {args.mapping}, seed={args.seed}")
    print(ascii_image(original, mapping=args.mapping))
    for domain in DOMAINS:
        coeff = transform_images(mask, domain=domain)
        recovered = inverse_transform(coeff, domain=domain)
        error = float(np.max(np.abs(recovered - mask)))
        print(f"[PASS] {domain:7s} max reconstruction error={error:.3g}")
        if error > 1e-10:
            raise AssertionError("spectral reconstruction tolerance exceeded")
    print(f"[PASS] Decoded main six: {decoded.tolist()}")


if __name__ == "__main__":
    main()
