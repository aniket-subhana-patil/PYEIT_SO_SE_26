"""
eit_helpers.py
==============
Shared helper functions for the EIT project notebooks.

Everything the notebooks need lives here: building meshes, placing objects,
simulating measurements, running the three reconstruction methods, scoring the
resulting images and plotting them.  Keeping it in one file means all five
notebooks use exactly the same settings.

Built on pyEIT: https://github.com/eitcom/pyEIT
"""

from __future__ import annotations

import textwrap
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import LinearTriInterpolator, Triangulation
from scipy.interpolate import RegularGridInterpolator

import pyeit.eit.bp as back_projection
import pyeit.eit.greit as greit_method
import pyeit.eit.jac as gauss_newton
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from pyeit.eit.fem import EITForward
from pyeit.eit.interp2d import sim2pts
from pyeit.mesh.external import place_electrodes_equal_spacing
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from pyeit.quality.merit import (calc_greit_figures_of_merit,
                                 calc_position_error)


# ---------------------------------------------------------------------------
# Version check
# ---------------------------------------------------------------------------
# The version of pyEIT on PyPI is much older than the one on GitHub and has a
# different scoring API.  Fail early with a clear message instead of a
# confusing error halfway through a notebook.
def _check_pyeit_version() -> None:
    import inspect
    missing = []
    if "conductive_target" not in inspect.signature(
            calc_greit_figures_of_merit).parameters:
        missing.append("calc_greit_figures_of_merit(..., conductive_target=)")
    if "method" not in inspect.signature(calc_position_error).parameters:
        missing.append("calc_position_error(..., method=)")
    if missing:
        raise ImportError(
            "This project needs the GitHub version of pyEIT, not the one on PyPI.\n"
            "Missing: " + ", ".join(missing) + "\n\n"
            "    git clone https://github.com/eitcom/pyEIT.git\n"
            "    pip install 'setuptools<60' wheel\n"
            "    pip install --no-build-isolation ./pyEIT\n\n"
            "or set the PYEIT_PATH environment variable to a source checkout."
        )


_check_pyeit_version()


# ---------------------------------------------------------------------------
# Settings used by every notebook
# ---------------------------------------------------------------------------

NUMBER_OF_ELECTRODES = 16
BACKGROUND_CONDUCTIVITY = 1.0

# Mesh spacing.  We simulate the measurements on a fine mesh and reconstruct on
# a coarser one, so the reconstruction cannot simply "cheat" by using the same
# grid that produced the data.
FINE_MESH_SPACING = 0.045
COARSE_MESH_SPACING = 0.08

IMAGE_SIZE = 64          # reconstructions are compared on a 64 x 64 picture
RANDOM_SEED = 2026       # fixed so every run gives the same numbers


# The three current injection patterns, described by how far apart the two
# current-carrying electrodes are.  Measurement is always between neighbouring
# electrodes so that only the injection pattern changes.
#
# "skip-4" means 4 electrodes are skipped between the pair, so the gap is 5.
INJECTION_PATTERNS = {
    "adjacent": 1,
    "opposite": NUMBER_OF_ELECTRODES // 2,
    "skip_4":   5,
}

PATTERN_NAMES = {
    "adjacent": "adjacent (gap 1)",
    "opposite": "opposite (gap 8)",
    "skip_4":   "skip-4 (gap 5)",
}

# The three reconstruction methods pyEIT provides.
METHODS = ("back_projection", "gauss_newton", "greit")

METHOD_NAMES = {
    "back_projection": "Back-projection",
    "gauss_newton":    "Gauss-Newton",
    "greit":           "GREIT",
}

METHOD_SHORT = {
    "back_projection": "BP",
    "gauss_newton":    "JAC",
    "greit":           "GREIT",
}

METHOD_COLOURS = {
    "back_projection": "#4c72b0",
    "gauss_newton":    "#dd8452",
    "greit":           "#55a868",
}

PATTERN_COLOURS = {
    "adjacent": "#4c72b0",
    "opposite": "#c44e52",
    "skip_4":   "#8172b3",
}

# Default settings for each method, taken from the pyEIT examples.
METHOD_SETTINGS = {
    "back_projection": dict(weight="none"),
    "gauss_newton":    dict(p=0.5, lamb=0.03, method="kotre", perm=1,
                            jac_normalized=True),
    "greit":           dict(p=0.5, lamb=0.01, n=IMAGE_SIZE, s=20.0, ratio=0.1,
                            perm=1, jac_normalized=True),
}

COLOUR_MAP = "RdBu_r"        # red = more conductive, blue = less


def use_project_style() -> None:
    """Apply the plot style used for all figures in this project."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "image.cmap": COLOUR_MAP,
    })


# ---------------------------------------------------------------------------
# Meshes and objects
# ---------------------------------------------------------------------------

def build_meshes(number_of_electrodes: int = NUMBER_OF_ELECTRODES):
    """
    Build the two meshes used throughout the project.

    Returns
    -------
    fine_mesh   : used to simulate the measurements
    coarse_mesh : used to reconstruct the image
    """
    fine_mesh = mesh.create(number_of_electrodes, h0=FINE_MESH_SPACING)
    fine_mesh.el_pos = np.array(place_electrodes_equal_spacing(
        fine_mesh, n_electrodes=number_of_electrodes))

    coarse_mesh = mesh.create(number_of_electrodes, h0=COARSE_MESH_SPACING)
    coarse_mesh.el_pos = np.array(place_electrodes_equal_spacing(
        coarse_mesh, n_electrodes=number_of_electrodes))

    return fine_mesh, coarse_mesh


def add_objects(fine_mesh, objects: Sequence[dict],
                background: float = BACKGROUND_CONDUCTIVITY):
    """
    Place circular objects into the mesh.

    Each object is a dict: {"centre": [x, y], "radius": r, "conductivity": c}.
    A conductivity above the background is more conductive, below is less.
    """
    circles = [PyEITAnomaly_Circle(center=list(item["centre"]),
                                   r=float(item["radius"]),
                                   perm=float(item["conductivity"]))
               for item in objects]
    return mesh.set_perm(fine_mesh, anomaly=circles, background=background)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def build_measurement_setup(pattern: str = "adjacent",
                            number_of_electrodes: int = NUMBER_OF_ELECTRODES):
    """
    Build the measurement plan: which electrode pairs inject current and which
    pairs are read out.  `pattern` is one of INJECTION_PATTERNS, or an integer
    electrode gap.
    """
    if isinstance(pattern, str):
        if pattern not in INJECTION_PATTERNS:
            raise KeyError(f"unknown pattern {pattern!r}; "
                           f"pick one of {list(INJECTION_PATTERNS)}")
        gap = INJECTION_PATTERNS[pattern]
    else:
        gap = int(pattern)

    return protocol.create(number_of_electrodes, dist_exc=gap,
                           step_meas=1, parser_meas="std")


def simulate_measurements(fine_mesh, mesh_with_objects, measurement_setup):
    """
    Simulate the two sets of boundary voltages that difference imaging needs.

    Returns
    -------
    background_voltages : with nothing in the tank
    object_voltages     : with the objects present
    """
    simulator = EITForward(fine_mesh, measurement_setup)
    background_voltages = simulator.solve_eit(perm=fine_mesh.perm)
    object_voltages = simulator.solve_eit(perm=mesh_with_objects.perm)
    return background_voltages, object_voltages


def add_measurement_noise(background_voltages, object_voltages,
                          signal_to_noise_db: Optional[float],
                          random_generator=None):
    """
    Add random noise to both sets of voltages.

    The noise level is set relative to the size of the measured voltages:
    a higher signal-to-noise number means cleaner data.  Real EIT hardware
    manages roughly 60-80 dB.
    """
    if signal_to_noise_db is None:
        return background_voltages, object_voltages

    if random_generator is None:
        random_generator = np.random.default_rng(RANDOM_SEED)

    typical_size = float(np.sqrt(np.mean(np.abs(background_voltages) ** 2)))
    noise_level = typical_size / (10.0 ** (signal_to_noise_db / 20.0))

    return (background_voltages + random_generator.normal(
                0.0, noise_level, size=background_voltages.shape),
            object_voltages + random_generator.normal(
                0.0, noise_level, size=object_voltages.shape))


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

@dataclass
class Reconstruction:
    """One reconstructed image, plus how long it took to produce."""
    method: str
    image: np.ndarray                     # 64 x 64, NaN outside the circle
    setup_seconds: float = 0.0            # cost of preparing the method
    solve_seconds: float = 0.0            # cost of one frame
    settings: dict = field(default_factory=dict)

    @property
    def scaled(self) -> np.ndarray:
        """Image divided by its own peak, so different methods compare fairly."""
        peak = np.nanmax(np.abs(self.image))
        return self.image / peak if peak > 0 else self.image


def prepare_method(method: str, coarse_mesh, measurement_setup,
                   settings: Optional[dict] = None):
    """
    Set up one reconstruction method.  This is the expensive part and only has
    to be done once per mesh and measurement plan, so the notebooks reuse the
    result across many frames.
    """
    options = dict(METHOD_SETTINGS[method])
    if settings:
        options.update(settings)

    started = time.perf_counter()
    if method == "back_projection":
        solver = back_projection.BP(coarse_mesh, measurement_setup)
    elif method == "gauss_newton":
        solver = gauss_newton.JAC(coarse_mesh, measurement_setup)
    elif method == "greit":
        solver = greit_method.GREIT(coarse_mesh, measurement_setup)
    else:
        raise KeyError(f"unknown method {method!r}; pick one of {METHODS}")
    solver.setup(**options)
    setup_seconds = time.perf_counter() - started

    return solver, setup_seconds, options


def reconstruct_image(method: str, coarse_mesh, measurement_setup,
                      object_voltages, background_voltages,
                      settings: Optional[dict] = None,
                      prepared_solver=None,
                      setup_seconds: Optional[float] = None) -> Reconstruction:
    """
    Turn a pair of voltage readings into an image.

    Pass `prepared_solver` (from `prepare_method`) to skip the setup cost when
    reconstructing many frames with the same method.
    """
    options = dict(METHOD_SETTINGS[method])
    if settings:
        options.update(settings)

    if prepared_solver is None:
        prepared_solver, setup_seconds, options = prepare_method(
            method, coarse_mesh, measurement_setup, settings)
    setup_seconds = 0.0 if setup_seconds is None else setup_seconds

    started = time.perf_counter()
    raw_result = prepared_solver.solve(object_voltages, background_voltages,
                                       normalize=True)
    solve_seconds = time.perf_counter() - started

    # The three methods return their answers in three different shapes, so each
    # one gets converted onto the same 64 x 64 picture.
    if method == "greit":
        x_grid, y_grid, pixel_grid = prepared_solver.mask_value(
            np.real(raw_result), mask_value=np.nan)
        image = _grid_to_image(x_grid, y_grid, pixel_grid)
    else:
        values = np.real(raw_result)
        if values.size == coarse_mesh.n_elems:       # Gauss-Newton: per triangle
            values = sim2pts(coarse_mesh.node, coarse_mesh.element, values)
        image = _mesh_to_image(coarse_mesh, values)  # back-projection: per corner

    return Reconstruction(method, image, setup_seconds, solve_seconds, options)


def reconstruct_with_all_methods(coarse_mesh, measurement_setup,
                                 object_voltages, background_voltages
                                 ) -> Dict[str, Reconstruction]:
    """Run all three methods on the same measurements."""
    return {method: reconstruct_image(method, coarse_mesh, measurement_setup,
                                      object_voltages, background_voltages)
            for method in METHODS}


# ---------------------------------------------------------------------------
# Putting everything onto one common picture
# ---------------------------------------------------------------------------
# Back-projection gives a value per mesh corner, Gauss-Newton one per triangle
# and GREIT its own pixel grid.  To compare them we redraw all three onto the
# same 64 x 64 picture covering the tank, with the area outside set to NaN.

_pixel_positions = np.linspace(-1.0, 1.0, IMAGE_SIZE)
_pixel_x, _pixel_y = np.meshgrid(_pixel_positions, _pixel_positions)
_inside_tank = (_pixel_x ** 2 + _pixel_y ** 2) <= 1.0

IMAGE_EXTENT = (-1.0, 1.0, -1.0, 1.0)


def _mesh_to_image(any_mesh, corner_values) -> np.ndarray:
    """Redraw values given at mesh corners onto the common picture."""
    triangles = Triangulation(any_mesh.node[:, 0], any_mesh.node[:, 1],
                              any_mesh.element)
    interpolate = LinearTriInterpolator(
        triangles, np.asarray(corner_values, dtype=float))
    picture = np.asarray(interpolate(_pixel_x, _pixel_y).filled(np.nan),
                         dtype=float)
    picture[~_inside_tank] = np.nan
    return picture


def triangle_values_to_image(any_mesh, triangle_values) -> np.ndarray:
    """Redraw values given per triangle onto the common picture."""
    corner_values = sim2pts(any_mesh.node, any_mesh.element,
                            np.asarray(triangle_values, float))
    return _mesh_to_image(any_mesh, corner_values)


def _grid_to_image(x_grid, y_grid, pixel_grid) -> np.ndarray:
    """Redraw a GREIT pixel grid onto the common picture."""
    grid = np.array(pixel_grid, dtype=float)
    blank = np.isnan(grid)

    read_values = RegularGridInterpolator(
        (y_grid[:, 0], x_grid[0, :]), np.where(blank, 0.0, grid),
        bounds_error=False, fill_value=np.nan)
    read_blanks = RegularGridInterpolator(
        (y_grid[:, 0], x_grid[0, :]), blank.astype(float),
        bounds_error=False, fill_value=1.0)

    wanted = np.stack([_pixel_y.ravel(), _pixel_x.ravel()], axis=-1)
    picture = read_values(wanted).reshape(_pixel_x.shape)
    picture[read_blanks(wanted).reshape(_pixel_x.shape) > 0.5] = np.nan
    picture[~_inside_tank] = np.nan
    return picture


def true_picture(fine_mesh, mesh_with_objects,
                 background: float = BACKGROUND_CONDUCTIVITY) -> np.ndarray:
    """
    The ground truth, drawn on the same picture as the reconstructions, as the
    change in conductivity from the background.  Sampled per triangle so the
    objects keep sharp edges.
    """
    change = np.real(mesh_with_objects.perm - fine_mesh.perm)
    triangles = Triangulation(fine_mesh.node[:, 0], fine_mesh.node[:, 1],
                              fine_mesh.element)
    which_triangle = triangles.get_trifinder()(_pixel_x, _pixel_y)

    picture = np.full(_pixel_x.shape, np.nan)
    found = which_triangle >= 0
    picture[found] = change[which_triangle[found]]
    picture[~_inside_tank] = np.nan
    return picture


# ---------------------------------------------------------------------------
# Scoring an image
# ---------------------------------------------------------------------------

SCORE_MEANINGS = {
    "size":            "how big the reconstructed blob is (0.15 = correct here)",
    "position_error":  "how far the blob sits from where the object really is",
    "shape_error":     "how far from circular the blob is (0 = perfect circle)",
    "ringing":         "how much false opposite-coloured halo surrounds the blob",
    "brightness":      "average pixel value of the image",
}


def score_image(truth: np.ndarray, reconstruction: np.ndarray,
                conductive_object: bool = True) -> Dict[str, float]:
    """
    Score a reconstruction against the truth using pyEIT's own image quality
    measures (the standard GREIT figures of merit).

    The reconstruction is divided by its peak first, because back-projection
    produces images on an arbitrary scale.
    """
    picture = np.asarray(reconstruction, dtype=float)
    peak = np.nanmax(np.abs(picture))
    if peak > 0:
        picture = picture / peak

    truth = np.asarray(truth, dtype=float)
    brightness, _, size, shape_error, ringing = calc_greit_figures_of_merit(
        truth, picture, conductive_target=conductive_object)

    position_error = calc_position_error(
        truth, picture, conductive_target=conductive_object, method="Euclidean")

    return {"size": float(size),
            "position_error": float(position_error),
            "shape_error": float(shape_error),
            "ringing": float(ringing),
            "brightness": float(brightness)}


def contrast_to_noise(reconstruction: np.ndarray, truth: np.ndarray) -> float:
    """
    How clearly the object stands out from the background: the difference
    between the average value on the object and the average value elsewhere,
    divided by how much the background wobbles.  Bigger is better; below about
    2 the object is hard to pick out.
    """
    picture = np.asarray(reconstruction, float)
    truth = np.asarray(truth, float)

    on_object = np.isfinite(truth) & (np.abs(truth) > 1e-9)
    on_background = np.isfinite(truth) & (np.abs(truth) <= 1e-9) & np.isfinite(picture)
    if on_object.sum() == 0 or on_background.sum() == 0:
        return np.nan

    background_wobble = np.nanstd(picture[on_background])
    if background_wobble == 0:
        return np.nan
    return float((np.nanmean(picture[on_object])
                  - np.nanmean(picture[on_background])) / background_wobble)


def sensitivity_map(coarse_mesh, measurement_setup) -> np.ndarray:
    """
    How strongly each part of the tank affects the measurements.  Areas with a
    low value are hard to see: a change there barely shows up at the electrodes.
    """
    simulator = EITForward(coarse_mesh, measurement_setup)
    sensitivity, _ = simulator.compute_jac(perm=coarse_mesh.perm, normalize=False)
    per_triangle = np.sqrt(np.sum(np.real(sensitivity) ** 2, axis=0))
    picture = triangle_values_to_image(coarse_mesh, per_triangle)
    return picture / np.nanmax(picture)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def show_image(axis, picture, title: str = "", colour_map: str = COLOUR_MAP,
               peak: Optional[float] = None, symmetric: bool = True):
    """Draw one picture with a sensible colour scale and the tank outline."""
    picture = np.asarray(picture, float)
    if peak is None:
        peak = np.nanmax(np.abs(picture))
        peak = 1.0 if (not np.isfinite(peak) or peak == 0) else peak

    drawn = axis.imshow(picture, origin="lower", extent=IMAGE_EXTENT,
                        cmap=colour_map, vmin=-peak if symmetric else np.nanmin(picture),
                        vmax=peak, interpolation="nearest")
    axis.add_patch(plt.Circle((0, 0), 1.0, fill=False, lw=0.8, color="0.35", zorder=5))
    axis.set_aspect("equal")
    axis.set_xticks([]); axis.set_yticks([])
    for edge in axis.spines.values():
        edge.set_visible(False)
    if title:
        axis.set_title(textwrap.fill(title, 26))
    return drawn


def show_images(pictures: Sequence[np.ndarray], titles: Sequence[str],
                heading: str = "", scale_each: bool = True,
                panel_size: float = 2.6):
    """Draw a row of pictures side by side.  The workhorse figure."""
    pictures = [np.asarray(p, float) for p in pictures]
    if scale_each:
        pictures = [p / np.nanmax(np.abs(p)) if np.nanmax(np.abs(p)) > 0 else p
                    for p in pictures]

    figure, axes = plt.subplots(
        1, len(pictures),
        figsize=(panel_size * len(pictures), panel_size + 0.9),
        constrained_layout=True)
    axes = np.atleast_1d(axes)

    drawn = None
    for axis, picture, title in zip(axes, pictures, titles):
        drawn = show_image(axis, picture, title=title)
    if heading:
        figure.suptitle(heading, fontsize=11)
    figure.colorbar(drawn, ax=axes.tolist(), fraction=0.025, pad=0.02,
                    label="conductivity change")
    return figure, axes


def mark_electrodes(axis, any_mesh, label: bool = True):
    """Put dots where the electrodes are, on a plot drawn in mesh coordinates."""
    positions = any_mesh.node[any_mesh.el_pos]
    axis.scatter(positions[:, 0], positions[:, 1], s=18, c="k", zorder=6,
                 clip_on=False)
    if label:
        for number, (x, y) in enumerate(positions[:, :2]):
            axis.annotate(str(number), xy=(1.13 * x, 1.13 * y), color="0.25",
                          fontsize=7, ha="center", va="center", zorder=6,
                          annotation_clip=False)


def show_mesh_values(axis, any_mesh, triangle_values, title: str = ""):
    """Draw values straight onto the triangles of a mesh."""
    values = np.real(np.asarray(triangle_values, float))
    peak = np.nanmax(np.abs(values))
    peak = 1.0 if (not np.isfinite(peak) or peak == 0) else peak
    drawn = axis.tripcolor(any_mesh.node[:, 0], any_mesh.node[:, 1],
                           any_mesh.element, values, shading="flat",
                           cmap=COLOUR_MAP, vmin=-peak, vmax=peak)
    axis.set_aspect("equal")
    axis.set_xticks([]); axis.set_yticks([])
    for edge in axis.spines.values():
        edge.set_visible(False)
    if title:
        axis.set_title(textwrap.fill(title, 26))
    return drawn


use_project_style()
