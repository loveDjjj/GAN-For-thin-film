import numpy as np
import os


def compute_q_for_spectrum(wavelengths, absorption, center, half_window):
    """
    Compute Q-factor within [center - half_window, center + half_window].
    If no data/peak in that window, return q=0.
    """
    window_mask = (wavelengths >= center - half_window) & (wavelengths <= center + half_window)
    indices_in_window = np.where(window_mask)[0]
    if indices_in_window.size == 0:
        return {
            "q": 0.0,
            "peak_wavelength": None,
            "peak_absorption": None,
            "left_wavelength": None,
            "right_wavelength": None,
        }

    # Peak search in window
    peak_idx = indices_in_window[np.argmax(absorption[indices_in_window])]
    peak_wavelength = wavelengths[peak_idx]
    peak_absorption = absorption[peak_idx]

    # If peak_absorption is non-positive, treat as invalid
    if peak_absorption <= 0:
        return {
            "q": 0.0,
            "peak_wavelength": peak_wavelength,
            "peak_absorption": peak_absorption,
            "left_wavelength": None,
            "right_wavelength": None,
        }

    half_max = peak_absorption / 2.0

    # Find left half-max crossing
    left_idx = None
    for i in range(peak_idx, -1, -1):
        if absorption[i] <= half_max:
            left_idx = i
            break

    # Find right half-max crossing
    right_idx = None
    for i in range(peak_idx, len(wavelengths)):
        if absorption[i] <= half_max:
            right_idx = i
            break

    if left_idx is None or right_idx is None or right_idx == left_idx:
        return {
            "q": 0.0,
            "peak_wavelength": peak_wavelength,
            "peak_absorption": peak_absorption,
            "left_wavelength": None,
            "right_wavelength": None,
        }

    # Linear interpolation for more accurate half-max points
    if left_idx < peak_idx:
        left_wavelength = wavelengths[left_idx] + (wavelengths[left_idx + 1] - wavelengths[left_idx]) * (
            (half_max - absorption[left_idx]) / (absorption[left_idx + 1] - absorption[left_idx] + 1e-12)
        )
    else:
        left_wavelength = wavelengths[left_idx]

    if right_idx > peak_idx:
        right_wavelength = wavelengths[right_idx - 1] + (wavelengths[right_idx] - wavelengths[right_idx - 1]) * (
            (half_max - absorption[right_idx - 1]) / (absorption[right_idx] - absorption[right_idx - 1] + 1e-12)
        )
    else:
        right_wavelength = wavelengths[right_idx]

    fwhm = right_wavelength - left_wavelength
    q = peak_wavelength / fwhm if fwhm > 0 else 0.0

    return {
        "q": float(q),
        "peak_wavelength": float(peak_wavelength),
        "peak_absorption": float(peak_absorption),
        "left_wavelength": float(left_wavelength),
        "right_wavelength": float(right_wavelength),
    }


def compute_q_for_indices(wavelengths, absorption_spectra, indices, center, half_window):
    """Compute Q for a list of sample indices."""
    results = []
    for idx in indices:
        q_info = compute_q_for_spectrum(wavelengths, absorption_spectra[idx], center, half_window)
        q_info["index"] = int(idx)
        results.append(q_info)
    return results


def save_q_report(path, q_results, title="Q Report"):
    """Save Q-factor results to a txt file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("=" * 60 + "\n\n")
        for i, res in enumerate(q_results, 1):
            f.write(f"Sample {i} (Original Index: {res.get('index', 'N/A')})\n")
            f.write(f"Q: {res['q']:.4f}\n")
            f.write(f"Peak Wavelength: {res.get('peak_wavelength')}\n")
        f.write(f"Peak Absorption: {res.get('peak_absorption')}\n")
        f.write(f"Left Half-Max: {res.get('left_wavelength')}\n")
        f.write(f"Right Half-Max: {res.get('right_wavelength')}\n")
        f.write("-" * 40 + "\n")
