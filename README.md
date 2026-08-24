# Investigation of Reconstruction Algorithms and Current Injection Patterns in EIT

Project Seminar Virtual Acoustics · supervisor M.Sc. Jacob P. Thönes · due 2026-09-25

Simulation study using [pyEIT](https://github.com/eitcom/pyEIT), looking at how the choice of
reconstruction method, the position and number of objects, and the current injection pattern affect
the images that come out.

> **First iteration.** The aim here was to get every part of the task covered end to end with
> readable code and honest observations, not to squeeze out final numbers. Things I would tighten
> next are listed at the bottom.

## The notebooks

| notebook | task | what it does |
|---|---|---|
| `01_getting_started.ipynb` | 1 | meshes, current paths, one simulated measurement, first reconstruction |
| `02_reconstruction_methods.ipynb` | 2 | back-projection vs Gauss-Newton vs GREIT: images, scores, timing, noise |
| `03_object_position.ipynb` | 3 | one object moved from the centre out to the edge |
| `04_number_of_objects.ipynb` | 4 | one to four objects, how close two can get, mixed types, size accuracy |
| `05_injection_patterns.ipynb` | 5, 6 | adjacent vs opposite vs skip-4, plus the overall summary |

`eit_helpers.py` holds everything shared — meshes, measurements, the three methods, scoring and
plotting — so all five notebooks use identical settings.

Notebooks are saved **with their outputs**, so the whole study can be read without running anything.
Figures go to `figures/`, tables to `results/`.

## Running it

> **Use the GitHub version of pyEIT, not `pip install pyeit`.** The version on PyPI is much older
> and its image-scoring functions have a different interface. `eit_helpers.py` checks this on
> import and tells you if the wrong one is installed.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

git clone https://github.com/eitcom/pyEIT.git
pip install "setuptools<60" wheel                      # newer setuptools breaks pyEIT's build
pip install --no-build-isolation ./pyEIT

pip install ipykernel jupyterlab
python -m ipykernel install --user --name pyeit --display-name "Python (pyeit)"
```

Then `jupyter lab`, choose the **Python (pyeit)** kernel, and run the notebooks in order from the
repository root. Everything runs in a few minutes; the random seed is fixed so the numbers repeat
exactly.

If you would rather not install pyEIT, clone it and set `PYEIT_PATH` to the checkout instead — the
first cell of every notebook picks that up.

## What came out of it

**The three methods**

| | back-projection | Gauss-Newton | GREIT |
|---|---|---|---|
| blob size (real radius 0.15) | 0.49 | **0.29** | 0.36 |
| position error | 0.042 | **0.012** | **0.012** |
| false halo (ringing) | **0.00** | 0.25 | 0.10 |
| visibility, clean data | 3.3 | **6.6** | 5.1 |
| visibility at 30 dB noise | 2.3 | 1.7 | **2.9** |
| setup time | **0.05 s** | 0.26 s | 1.00 s |

Gauss-Newton is the sharpest on clean data but the first to fall apart under noise. GREIT is
slightly blurrier but rounder, calmer and much more robust — the best default. Back-projection is
cheap and never misbehaves, but its blob is three times too big.

**Position.** The middle of the tank is about 23× less visible than the edge. Objects near the edge
give roughly three times more signal, reconstruct tighter and survive noise better. Under noise a
blind spot opens up in the centre and grows outward.

**Number of objects.** Two objects merge into one blob once their centres are closer than about
twice their diameter. Adding objects does not weaken each one, but the halos clutter the background
so everything gets harder to pick out. Conductive and non-conductive objects are never confused.
Reconstructed size always overstates the truth, worst for small objects (up to 8× at radius 0.06).

**Injection patterns.** Opposite and skip-4 have far more even sensitivity (centre 10–12× less
visible than the edge, against 23× for adjacent) and produce several times more signal for a central
object. Despite that, **adjacent injection gave the sharpest images at every position here** — its
16 injections carry more independent information, while the opposite ones largely repeat each other.

## What I would do differently next time

- **The noise comparison between patterns is the weak point.** The noise is scaled to each pattern's
  own voltage level, which cancels out the fact that opposite and skip-4 produce roughly 5× larger
  voltages. Real hardware has a fixed noise floor, so those patterns would do better than shown
  here. There is already a hint of it — at 30 dB with the object near the edge, opposite overtakes
  adjacent. Redoing this with a fixed noise floor is the first thing to fix.
- Only 5 noise runs per level, so the noisiest points are still a bit rough.
- No check of how sensitive the results are to the smoothing settings of Gauss-Newton and GREIT.
- Everything is 2-D, electrodes are treated as points, and the tank shape is assumed known exactly.

## Reference

Benyuan Liu et al., "pyEIT: A python based framework for Electrical Impedance Tomography",
*SoftwareX* 7 (2018), 304–308.
