# Author: Austin Seamann
# Last Update: 5-5-2026

import os
import sys
import argparse
import multiprocessing

import numpy as np
from pyrosetta import *
init(silent=True)

VERBOSE = False


def net_charge(seq, add_histidine=False):
    '''
    Calculate the net charge of a protein sequence
    '''
    charge = seq.count('K') + seq.count('R') - seq.count('D') - seq.count('E')
    if add_histidine:
        charge += seq.count('H')
    return charge


def index_selector(indices):
    '''
    Creates an index selector for a given selection. If a string is given,
    uses that string directly in the selector. Converts integers to strings for
    selector input. Converts ranges or lists of integers to comma-separated
    strings for selector input. - Adapted from Joey Utils
    '''
    from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector

    if isinstance(indices, str):
        return ResidueIndexSelector(indices)
    if isinstance(indices, int):
        return ResidueIndexSelector(str(indices))
    if type(indices) in [list, range]:
        ind_str = ','.join([str(i) for i in indices])
        return ResidueIndexSelector(ind_str)


def empty_selector():
    '''Returns a selector that selects nothing. - Adapted from Joey Utils'''
    from pyrosetta.rosetta.core.select.residue_selector import FalseResidueSelector
    return FalseResidueSelector()


def full_selector():
    '''Returns a selector that selects the entire pose. - Adapted from Joey Utils'''
    from pyrosetta.rosetta.core.select.residue_selector import TrueResidueSelector
    return TrueResidueSelector()


def not_selector(selection):
    '''Returns all residues not in the given selector. - Adapted from Joey Utils'''
    from pyrosetta.rosetta.core.select.residue_selector import NotResidueSelector
    return NotResidueSelector(selection)


def selector_intersection(*selectors):
    '''Returns the intersection of two or more selectors. - Adapted from Joey Utils'''
    from pyrosetta.rosetta.core.select.residue_selector import AndResidueSelector
    result = AndResidueSelector()
    for s in selectors:
        result.add_residue_selector(s)
    return result


def selector_union(*selectors):
    '''Returns the union of two or more selectors. - Adapted from Joey Utils'''
    from pyrosetta.rosetta.core.select.residue_selector import OrResidueSelector
    result = OrResidueSelector()
    for s in selectors:
        result.add_residue_selector(s)
    return result


def intergroup_selector(selector_1, selector_2, nearby_atom=5.5, cb_dist=11.0,
                        vector_angle=75.0, vector_dist=9.0):
    '''
    Returns an InterGroupInterfaceByVectorSelector covering the shell of
    residues around the interface between two groups. - Adapted from Joey Utils
    '''
    from pyrosetta.rosetta.core.select.residue_selector import InterGroupInterfaceByVectorSelector
    sel = InterGroupInterfaceByVectorSelector()
    sel.group1_selector(selector_1)
    sel.group2_selector(selector_2)
    sel.nearby_atom_cut(nearby_atom)
    sel.cb_dist_cut(cb_dist)
    sel.vector_angle_cut(vector_angle)
    sel.vector_dist_cut(vector_dist)
    return sel


def make_move_map(bb=False, chi=False, jump=False):
    '''
    Creates a MoveMap. Arguments can be booleans (apply to all residues) or
    lists of residue numbers (apply only to those residues). - Adapted from Joey Utils
    '''
    from pyrosetta import MoveMap
    mm = MoveMap()
    if isinstance(bb, bool):
        mm.set_bb(bb)
    else:
        for res in bb:
            mm.set_bb(res, True)
    if isinstance(chi, bool):
        mm.set_chi(chi)
    else:
        for res in chi:
            mm.set_chi(res, True)
    if isinstance(jump, bool):
        mm.set_jump(jump)
    else:
        for j in jump:
            mm.set_jump(j, True)
    return mm


def make_task_factory(design_selection=None, repack_selection=None,
                      immobile_selection=None, res_changes=None, ex12=True):
    '''
    Builds a TaskFactory for use with movers. res_changes is a dict mapping
    pose-numbered residue indices to 1-letter amino acid codes to force.
    - Adapted from Joey Utils
    '''
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    import pyrosetta.rosetta.core.pack.task.operation as taskop

    if design_selection is None:
        design_selection = empty_selector()
    if repack_selection is None:
        repack_selection = full_selector()
    if immobile_selection is None:
        immobile_selection = empty_selector()

    tf = TaskFactory()
    tf.push_back(taskop.IncludeCurrent())
    tf.push_back(taskop.NoRepackDisulfides())

    if ex12:
        tf.push_back(taskop.ExtraRotamers(0, 1, 1))
        tf.push_back(taskop.ExtraRotamers(0, 2, 1))

    if res_changes:
        for site, aa in res_changes.items():
            res_sel = index_selector(str(site))
            restriction = taskop.RestrictAbsentCanonicalAASRLT()
            restriction.aas_to_keep(aa.upper())
            tf.push_back(taskop.OperateOnResidueSubset(restriction, res_sel))
            design_selection = selector_union(design_selection, res_sel)

    design_selection = selector_intersection(design_selection,
                                             not_selector(immobile_selection))
    repack_selection = selector_intersection(repack_selection,
                                             not_selector(immobile_selection))
    repack_selection = selector_intersection(repack_selection,
                                             not_selector(design_selection))
    immobile_selection = not_selector(selector_union(design_selection, repack_selection))

    restrict = taskop.RestrictToRepackingRLT()
    tf.push_back(taskop.OperateOnResidueSubset(restrict, repack_selection))
    prevent = taskop.PreventRepackingRLT()
    tf.push_back(taskop.OperateOnResidueSubset(prevent, immobile_selection))

    return tf


def fast_relax_mover(score_function=None, task_factory=None, movemap=None, repeats=5):
    '''
    Creates a FastRelax mover. Uses ref2015 with constraints by default.
    - Adapted from Joey Utils
    '''
    from pyrosetta.rosetta.protocols.relax import FastRelax
    fr = FastRelax(repeats)
    if score_function is None:
        score_function = get_fa_scorefxn()
    fr.set_scorefxn(score_function)
    if task_factory:
        fr.set_task_factory(task_factory)
    if movemap:
        fr.set_movemap(movemap)
    return fr


def total_energy(pose, score_function):
    '''
    Calculates total energy of a pose using TotalEnergyMetric. - Adapted from Joey Utils
    '''
    from pyrosetta.rosetta.core.simple_metrics.metrics import TotalEnergyMetric
    tem = TotalEnergyMetric()
    tem.set_scorefunction(score_function)
    return tem.calculate(pose)


def thread_sequence(pose, design_seq, og_seq, output_path, num_relax=5, verbose=None):
    '''
    Thread a designed sequence onto a backbone structure using PyRosetta FastRelax.
    Identifies mutations between og_seq and design_seq, then builds a constrained
    FastRelax that forces only the mutated residues and repacks the surrounding shell.
    The lowest-energy structure across num_relax runs is saved to output_path.
    Returns (best_energy, mutation_list) where mutation_list is a list of
    'WTposDesign' strings (e.g. 'A5K').
    verbose defaults to the global VERBOSE flag when not explicitly provided.
    '''
    global VERBOSE
    if verbose is None:
        verbose = VERBOSE
    sfxn = get_fa_scorefxn()

    # Identify mutated and native positions (1-indexed, stop at '/' delimiter)
    mutant_index_list = []
    mutation_list = []
    native_index_list = []
    mutation_labels = []
    for i, (wt_aa, des_aa) in enumerate(zip(og_seq, design_seq)):
        if des_aa == '/':
            break
        if des_aa != wt_aa:
            mutant_index_list.append(i + 1)
            mutation_list.append(des_aa)
            mutation_labels.append(f'{wt_aa}{i + 1}{des_aa}')
        else:
            native_index_list.append(i + 1)

    if verbose:
        print(f"  Threading {len(mutant_index_list)} mutation(s) onto {output_path}")
        if mutation_labels:
            print(f"  Mutations: {', '.join(mutation_labels)}")

    # If no mutations, just write the WT backbone
    if not mutant_index_list:
        pose.dump_pdb(output_path)
        sfxn(pose)
        energy = total_energy(pose, sfxn)
        return energy, mutation_labels

    mut_dict = dict(zip(mutant_index_list, mutation_list))
    mutated_sel = index_selector(mutant_index_list)
    native_sel = index_selector(native_index_list)
    shell_sel = intergroup_selector(mutated_sel, native_sel)

    # Collect shell residue indices for the movemap
    shell_bools = shell_sel.apply(pose)
    shell_index_list = [i for i, x in enumerate(shell_bools, 1) if x]

    mv = make_move_map(bb=shell_index_list, chi=shell_index_list, jump=shell_index_list)
    tf_mutant = make_task_factory(None, shell_sel, None, mut_dict, ex12=False)

    best_energy = float('inf')
    best_pose = None

    for run in range(num_relax):
        work_pose = Pose()
        work_pose.detached_copy(pose)

        # Constrained relax with forced mutations
        fr_mutant = fast_relax_mover(score_function=sfxn, task_factory=tf_mutant, movemap=mv)
        fr_mutant.apply(work_pose)

        # Full final relaxation to resolve any clashes
        fr_full = fast_relax_mover(score_function=sfxn)
        fr_full.apply(work_pose)

        energy = total_energy(work_pose, sfxn)
        if energy < best_energy:
            best_energy = energy
            best_pose = work_pose

        if verbose:
            print(f"  Run {run + 1}/{num_relax}: energy = {energy:.3f}", flush=True)

    best_pose.dump_pdb(output_path)
    return best_energy, mutation_labels


def _thread_worker(task):
    '''
    Top-level worker for multiprocessing-based threading.
    Each worker spawns its own PyRosetta instance (required — PyRosetta cannot
    be shared across forked processes) and loads the backbone from disk.
    Returns (seq_id, energy, mutation_labels, output_path).
    '''
    pdb_path, seq_id, design_seq, og_seq, output_path, num_relax, verbose = task
    from pyrosetta import init as pyrosetta_init, pose_from_pdb, Pose
    pyrosetta_init(silent=True)
    pose = pose_from_pdb(pdb_path)
    energy, mutation_labels = thread_sequence(pose, design_seq, og_seq, output_path, num_relax, verbose)
    print(f'  Wrote {output_path}  (energy={energy:.3f})', flush=True)
    return seq_id, energy, mutation_labels, output_path


def parse_fasta_file(fasta_path):
    '''
    Parse a FASTA file. Returns a list of (header, sequence) tuples in order.
    '''
    entries = []
    header = None
    seq_parts = []
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    entries.append((header, ''.join(seq_parts)))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        entries.append((header, ''.join(seq_parts)))
    return entries


def format_mutation_list(mutation_labels, max_show=10):
    '''Format a list of mutation labels for display, truncating if too long.'''
    if not mutation_labels:
        return '(none)'
    if len(mutation_labels) <= max_show:
        return ', '.join(mutation_labels)
    return ', '.join(mutation_labels[:max_show]) + f' ... (+{len(mutation_labels) - max_show} more)'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Thread sequences from a FASTA file onto a PDB backbone using PyRosetta FastRelax.'
    )
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input PDB file (backbone to thread onto)')
    parser.add_argument('-f', '--fasta', type=str, required=True,
                        help='Input FASTA file of designed sequences')
    parser.add_argument('-o', '--output_dir', type=str, default='threaded',
                        help='Output directory for threaded PDB structures (default: threaded)')
    parser.add_argument('--num_relax', type=int, default=5,
                        help='Number of FastRelax runs per sequence; lowest-energy structure is kept (default: 5)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel worker processes (default: 8). '
                             'Each worker spawns an independent PyRosetta process.')
    parser.add_argument('--skip_wt', action='store_true', default=False,
                        help='Skip sequences that are identical to the WT (no mutations)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output (print per-run energies and full mutation lists)')
    return parser.parse_args()


def main(args):
    global VERBOSE
    if args.verbose:
        VERBOSE = True

    pdb_path = os.path.abspath(args.input)
    fasta_path = os.path.abspath(args.fasta)

    if not os.path.isfile(pdb_path):
        print(f"ERROR: PDB file not found: {pdb_path}")
        sys.exit(1)
    if not os.path.isfile(fasta_path):
        print(f"ERROR: FASTA file not found: {fasta_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load backbone and get WT sequence
    print(f"Loading backbone from {pdb_path} ...", flush=True)
    pose = pose_from_pdb(pdb_path)
    wt_seq = pose.sequence()
    wt_charge = net_charge(wt_seq)
    pdb_id = os.path.basename(pdb_path).replace('.pdb', '')

    print(f"  WT length : {len(wt_seq)} residues")
    print(f"  WT charge : {wt_charge:+d}")
    if VERBOSE:
        print(f"  WT sequence: {wt_seq}")

    # Parse FASTA
    entries = parse_fasta_file(fasta_path)
    print(f"\nFound {len(entries)} sequence(s) in {fasta_path}", flush=True)

    # Build task list
    tasks = []
    skipped = []
    for header, seq in entries:
        seq_id = header.split(',')[0]  # use the part before the first comma as identifier

        # Trim at '/' (chain delimiter used by ProteinMPNN)
        trimmed = seq.split('/')[0]

        # Align to WT — check lengths match
        compare_len = min(len(trimmed), len(wt_seq))
        mutations = [(i + 1, wt_seq[i], trimmed[i])
                     for i in range(compare_len) if trimmed[i] != wt_seq[i]]

        if args.skip_wt and not mutations:
            skipped.append(seq_id)
            if VERBOSE:
                print(f"  Skipping {seq_id} — identical to WT")
            continue

        out_name = os.path.join(args.output_dir, f'{seq_id}.pdb')
        tasks.append((pdb_path, seq_id, trimmed, wt_seq, out_name, args.num_relax, VERBOSE))

    if skipped:
        print(f"  Skipped {len(skipped)} WT-identical sequence(s): {', '.join(skipped)}")

    if not tasks:
        print("No sequences to thread. Exiting.")
        sys.exit(0)

    print(f"\nThreading {len(tasks)} sequence(s) using {min(args.workers, len(tasks))} worker(s) "
          f"({args.num_relax} relax run(s) each) ...\n", flush=True)

    # --- Run threading ---
    results = []  # list of (seq_id, energy, mutation_labels, output_path)

    num_workers = min(args.workers, len(tasks))
    if num_workers > 1:
        # Use 'spawn' — PyRosetta cannot be safely shared across forked processes;
        # each spawned worker initialises its own instance.
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=num_workers) as pool:
            results = pool.map(_thread_worker, tasks)
    else:
        # Single-worker path: reuse the already-initialised PyRosetta instance.
        for task in tasks:
            _, seq_id, design_seq, og_seq, out_name, num_relax, verbose = task
            energy, mutation_labels = thread_sequence(pose, design_seq, og_seq, out_name, num_relax, verbose)
            print(f'  Wrote {out_name}  (energy={energy:.3f})', flush=True)
            results.append((seq_id, energy, mutation_labels, out_name))

    # --- Summary table ---
    print(f"\n{'='*80}")
    print(f"  Threading Summary  —  backbone: {pdb_id}  |  {len(results)} sequence(s) threaded")
    print(f"{'='*80}")
    print(f"  {'Sequence ID':<30}  {'#Mut':>5}  {'Charge':>7}  {'Energy':>12}  Mutations")
    print(f"  {'-'*76}")

    # Print WT row for reference
    print(f"  {'[WT] ' + pdb_id:<30}  {'0':>5}  {wt_charge:>+7d}  {'N/A':>12}  (reference backbone)")

    for seq_id, energy, mutation_labels, out_name in results:
        # Reconstruct designed sequence from task list to calculate charge
        task_seq = next((t[2] for t in tasks if t[1] == seq_id), None)
        des_charge = net_charge(task_seq) if task_seq else '?'
        energy_str = f'{energy:.3f}' if isinstance(energy, float) else str(energy)
        mut_str = format_mutation_list(mutation_labels)
        print(f"  {seq_id:<30}  {len(mutation_labels):>5}  {des_charge:>+7d}  {energy_str:>12}  {mut_str}")

    print(f"{'='*80}")
    print(f"\nOutput PDB files written to: {os.path.abspath(args.output_dir)}/", flush=True)


if __name__ == '__main__':
    args = parse_args()
    main(args)
