#!/usr/bin/env python3
"""Minimal, guarded OpenJij change: TBTGF cache support for Dyson exchange only.
Run from the OpenJij repository root: python apply_dyson_tbtgf.py
"""
from pathlib import Path


def replace_once(text, old, new, label):
    found = text.count(old)
    if found != 1:
        raise RuntimeError(f"{label}: expected one source match, found {found}")
    return text.replace(old, new, 1)


def replace_n(text, old, new, n, label):
    found = text.count(old)
    if found != n:
        raise RuntimeError(f"{label}: expected {n} source matches, found {found}")
    return text.replace(old, new)


def updated_dyson(s):
    s = replace_once(s,
        '                             orbital_kernel: str = "full"):',
        '                             orbital_kernel: str = "full", self_energy_mode: str = "compute"):',
        'Dyson signature')
    s = replace_once(s,
        "'none' (only the integrated J_orb).\n    \"\"\"",
        "'none' (only the integrated J_orb).\n    self_energy_mode : 'compute', 'compute+write', or 'read' for electrode TBTGF I/O.\n    \"\"\"",
        'Dyson docstring')
    s = replace_once(s,
        '    size = comm.Get_size() if comm is not None else 1\n\n    z_array = np.asarray(energies, dtype=complex)',
        '''    size = comm.Get_size() if comm is not None else 1
    self_energy_mode = self_energy_mode.lower().strip()
    if self_energy_mode not in ("compute", "read", "compute+write"):
        raise ValueError("self_energy_mode must be compute, read or compute+write")
    if self_energy_mode != "compute" and calculation_folder is None:
        raise ValueError("calculation_folder required for TBTGF I/O")

    z_array = np.asarray(energies, dtype=complex)''',
        'Dyson mode validation')
    s = replace_once(s,
        '''    self_energy = JD.recursive_leads_from_hamiltonians(
        H_device=H, HamL=system.HamL, semi_L=system.semi_L, HamR=system.HamR,
        semi_R=system.semi_R, left_orbs=system.left_orbs, right_orbs=system.right_orbs,
        eta=lead_eta, polarized_leads=False)
    k0 = np.zeros(3)''',
        '''    k0 = np.zeros(3)
    if self_energy_mode == "compute":
        self_energy = JD.recursive_leads_from_hamiltonians(
            H_device=H, HamL=system.HamL, semi_L=system.semi_L, HamR=system.HamR,
            semi_R=system.semi_R, left_orbs=system.left_orbs, right_orbs=system.right_orbs,
            eta=lead_eta, polarized_leads=False)
    else:
        from exchange_mpi_io import TBTGFProvider, write_electrode_tbtgf
        se_folder = os.path.join(calculation_folder, "Self_Energies")
        if self_energy_mode == "compute+write":
            write_electrode_tbtgf(
                se_folder, H, system.HamL, system.semi_L, system.HamR, system.semi_R,
                z_array, nk=(1, 0, 0), left_orbs=system.left_orbs,
                right_orbs=system.right_orbs, lead_eta=lead_eta,
                mpi=comm is not None and size > 1)
        self_energy = TBTGFProvider(
            se_folder, H, system.HamL, system.HamR, system.semi_L, system.semi_R,
            z_array, k0[None, :], np.ones(1), left_orbs=system.left_orbs,
            right_orbs=system.right_orbs, lead_eta=lead_eta)''',
        'Dyson electrode setup')
    s = replace_once(s,
        '''    def propagators(z):
        """(G_ij, G_ji): spin up / down for LKAG, the same G for the rkky methods."""''',
        '''    def lead_terms(z, ie):
        if self_energy_mode == "compute":
            return self_energy(z, k0, 0)
        return self_energy(z, k0, 0, ie=ie, ik=0)

    def propagators(z, ie):
        """(G_ij, G_ji): spin up / down for LKAG, the same G for the rkky methods."""''',
        'Dyson lead access')
    s = replace_n(s,
        'JD._lead_embedding(n, self_energy(z, k0, 0))',
        'JD._lead_embedding(n, lead_terms(z, ie))', 2,
        'Dyson Green-function construction')
    s = replace_once(s,
        '        G_ij, G_ji = propagators(z)',
        '        G_ij, G_ji = propagators(z, ie)',
        'Dyson contour loop')
    s = replace_once(s,
        '        if rank != 0:\n            return None\n\n    J_k',
        '''        if rank != 0:
            if self_energy_mode != "compute":
                self_energy.close()
            return None

    if self_energy_mode != "compute":
        self_energy.close()

    J_k''',
        'Dyson TBTGF closure')
    return s


def updated_workflow(s):
    s = replace_once(s,
        'self-energies in folder/Self_Energies (calculate_exchange only)',
        'self-energies in folder/Self_Energies (all exchange methods)',
        'workflow docstring')
    s = replace_once(s,
        '            comm=comm if parallel else None, solver=solver, orbital_kernel=orbital_kernel)',
        '''            comm=comm if parallel else None, solver=solver, orbital_kernel=orbital_kernel,
            self_energy_mode=self_energy_mode,
            calculation_folder=folder if self_energy_mode != "compute" else None)''',
        'workflow Dyson call')
    return s


def main():
    files = {
        Path('Dyson_Exchange.py'): updated_dyson,
        Path('Exchange_Workflow.py'): updated_workflow,
    }
    updates = {}
    for path, update in files.items():
        original = path.read_text(encoding='utf-8')
        modified = update(original)
        compile(modified, str(path), 'exec')
        updates[path] = modified
    # Changes are written only after both files have passed all guards and syntax checks.
    for path, text in updates.items():
        path.write_text(text, encoding='utf-8')
        print(f'Updated {path}')
    print('No other files changed.')


if __name__ == '__main__':
    main()
