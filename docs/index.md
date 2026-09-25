# LAMTA

[![SPASSOv2.0 paper](https://img.shields.io/badge/SPASSOv2.0-paper-blue)](https://doi.org/10.1175/JTECH-D-24-0071.1)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

**LAMTA** (**LA**grangian **M**anifolds **T**racking **A**lgorithm) is a Python code designed to compute numerical particle trajectories within ocean current 2-D fields and to derive a range of Lagrangian diagnostics for detecting and tracking (sub)mesoscale ocean features.

---

## Manuscript and code

A dedicated manuscript is currently under development to describe the theoretical background, numerical methods, and applications of LAMTA in detail.

The source code is openly developed and maintained, with a focus on reproducible Lagrangian analyses in oceanography.

LAMTA is also described in the context of SPASSOv2.0 in:

Rousselet, L., d’Ovidio, F., Izard, L., Della Penna, A., Petrenko, A., Barrillon, S.,
Nencioli, F., & Doglioli, A. (2025). *A Software Package for an Adaptive
Satellite-Based Sampling for Oceanographic Cruises (SPASSOv2.0): Tracking
Fine-Scale Features for Physical and Biogeochemical Studies*. Journal of
Atmospheric and Oceanic Technology, 42(8), 979–990.
https://doi.org/10.1175/JTECH-D-24-0071.1

---

## Using LAMTA with the examples

Tutorials and example workflows are provided as Jupyter notebooks and are documented separately in the [**LAMTA Examples documentation**](https://lamta-examples.readthedocs.io/).

These examples are designed to run against a local installation of LAMTA and are not a standalone package. To run the tutorials make sure that:

- LAMTA is installed in the same Python environment (preferably in editable mode)

- that environment is selected as the active Python kernel (Jupyter, VS Code, etc.)

This setup ensures that the examples always run against your local version of LAMTA. The examples include:

- Initialising and advecting particles in analytical flows
- Working with `ParticleSet`
- Configuring advection schemes and parameters
- Handling periodic boundary conditions
- Applying LAMTA to realistic ocean current fields

---

## Installation

LAMTA is currently **not distributed as a packaged release on PyPI**. As a result, the command:

```bash
pip install lamta
```

will **not work** at this stage.

LAMTA must therefore be installed **from source**, either for local use or for development.

---

## Recommended installation (local / development)

This installation mode is recommended if you want to:

- run the LAMTA tutorials and examples
- explore or modify the code
- develop new diagnostics or workflows

The installation is performed in *editable* mode so that the local source tree is directly linked to your Python environment.

## Development environment

Recommended local setup (both repositories side-by-side):

```text
lamta_dev/
├─ LAMTA/           # core package
└─ LAMTA_examples/  # notebooks
```

This keeps the library and the examples aligned during development.

## VS Code

We recommend using Visual Studio Code and opening the parent folder (e.g. `lamta_dev/`) so both repositories are available in a single workspace. This makes it easy to edit the LAMTA source code and immediately test changes in the example notebooks, while keeping navigation and Git operations clear.

- [VS Code](https://code.visualstudio.com/)
- [VS Code workspaces](https://code.visualstudio.com/docs/editor/multi-root-workspaces)


### Step 1 — Clone the repositories

```{warning}
If you plan to contribute to the code or documentation, we recommend forking the corresponding repository first and cloning your fork instead.
```

```bash
# Users
git clone https://github.com/OceanCruises/LAMTA
git clone https://github.com/OceanCruises/LAMTA_examples

# Contributors
# git clone https://github.com/<your-username>/LAMTA
# git clone https://github.com/<your-username>/LAMTA_examples

cd LAMTA
```

### Step 2 — Create and activate a Python environment

Using **Conda** (recommended):

```bash
conda create -n lamta python=3.12
conda activate lamta
```

### Step 3 — Install LAMTA in editable mode

From the root of the cloned repository:

```bash
pip install -e .
```

This links LAMTA to your local source directory. Any modification to the code will be immediately reflected when importing LAMTA in Python, scripts, or notebooks.

---

## Future packaging

A standard packaged release (e.g. via PyPI) is planned for the future. Until then, installing LAMTA from source as described above is the supported and recommended approach.

## Funding and support

The development of LAMTA has been supported by the following organisations:

- **LOCEAN** — *Laboratoire d’Océanographie et du Climat: Expérimentations et Approches Numériques*, Sorbonne University
- **CNES** — *Centre National d’Études Spatiales*

## Acknowledgements

The development of LAMTA has benefited from the remarkable coding efforts and scientific contributions of:

- Louise Rousselet
- Francesco d’Ovidio
- Gina Fifani
