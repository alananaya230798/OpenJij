# OpenJij

**Open-boundary magnetic exchange interactions from Green's functions**

> **Status:** Early development. APIs, numerical conventions, and validation benchmarks are not yet finalized.

OpenJij is a Python research project for evaluating magnetic exchange interactions, especially between magnetic defects and spatially distributed magnetic moments in **graphene and carbon-based nanostructures**. The intended framework is applicable to other materials whenever a compatible electronic Hamiltonian and magnetic perturbations are available.

OpenJij explores complementary Green's-function formulations:

- **RKKY-type exchange:** magnetic perturbations coupled through a reference Green's function, with options to examine the choice of reference system and impurity scattering.
- **LKAG (Liechtenstein) exchange:** exchange interactions obtained from a spin-polarized reference Hamiltonian and infinitesimal rotations of magnetic moments.
- **Open boundaries:** embedding of a finite region into semi-infinite leads through electrode self-energies, avoiding artificial periodic images along the transport direction.

The objective is to connect first-principles Hamiltonians (initially from **SIESTA / TranSiesta**, with **sisl** as a Python interface) to exchange calculations and reproducible benchmarks. An important focus is how the definition of magnetic regions and orbital subspaces affects the resulting exchange parameters.

## Proposed repository layout

The module responsibilities below are **provisional** and should be adjusted after the existing scripts are added.

```text
OpenJij/
├── README.md
├── requirements.txt
├── requirements-optional.txt
├── .gitignore
├── Device_Builder.py          # Geometry / device / electrode preparation
├── Dyson_Exchange.py          # Green's functions, Dyson embedding or scattering methods
├── exchange_mpi_io.py        # MPI coordination, inputs / outputs, checkpointing
├── exchange_run.py           # Main exchange calculation entry point
├── Exchange_Workflow.py      # End-to-end preparation and analysis workflow (format TBD)
├── J_dependencies.py         # Shared exchange kernels and numerical helpers
├── plot_dependencies.py      # Plotting and post-processing
├── examples/
└── tests/
```

**Note:** The seven Python modules shown above are *expected files*, not included implementations. This starter archive does not invent or replace their contents; copy in your current validated versions. `Exchange_Workflow` may instead be a notebook or directory if that is how your project is organized.

## Installation

Use a Python environment with a working MPI installation if running parallel calculations.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For optional utilities:

```bash
python -m pip install -r requirements-optional.txt
```

On HPC systems, install `mpi4py` against the **same MPI implementation** used to launch the job. The exact supported Python, sisl, and SIESTA versions will be documented once the scripts have been checked.

## Workflow (planned)

1. Obtain reference and impurity Hamiltonians from SIESTA / TranSiesta.
2. Construct and verify the device–electrode geometry, orbital indexing, and Hamiltonian alignment.
3. Select magnetic regions, orbital subspaces, and spin-dependent perturbations.
4. Construct the required reference Green's functions, including lead self-energies where applicable.
5. Evaluate exchange integrals using the selected RKKY-type or LKAG expression.
6. Analyze orbital-resolved contributions and compare with independent benchmarks (e.g., total-energy differences or periodic TB2J).

The precise executable commands, input parameters and file formats will be documented from the working `exchange_run.py` implementation.

## Conventions and validation

Before publication, OpenJij should explicitly document:

- The spin Hamiltonian convention and the sign of ferromagnetic versus antiferromagnetic `J_ij`.
- Prefactors, whether a pair is counted once or twice, and whether moments are unit vectors or have explicit magnitudes.
- The definition of each local magnetic perturbation, including any spatially extended support.
- Nonorthogonal-basis treatment, orbital selection, device/lead alignment, and energy integration.
- Reference-state dependence, numerical convergence, and limits of the RKKY/LKAG approximations.

These conventions are **not assumed here**; they need to be verified against the implementation.

## Scientific background and attribution

OpenJij implements established exchange-interaction formalisms; it does not claim to introduce RKKY or LKAG theory. Useful starting references include:

1. A. I. Liechtenstein *et al.*, *Journal of Magnetism and Magnetic Materials* **67**, 65–74 (1987). [DOI](https://doi.org/10.1016/0304-8853(87)90721-9).
2. X. He, N. Helbig, M. J. Verstraete, and E. Bousquet, “TB2J: a python package for computing magnetic interaction parameters,” *Computer Physics Communications* **264**, 107938 (2021). [DOI](https://doi.org/10.1016/j.cpc.2021.107938).
3. A. Szilva *et al.*, “Quantitative theory of magnetic interactions in solids,” *Reviews of Modern Physics* **95**, 035004 (2023). [DOI](https://doi.org/10.1103/RevModPhys.95.035004).

Additional credits for sisl, SIESTA, TranSiesta, underlying methods, and reused code will be recorded when the relevant implementations are added.

## Project status

Initial README and dependency manifest only. Interfaces, installation tests, example inputs, benchmarks, licensing, authorship, and contribution guidelines remain to be defined.
