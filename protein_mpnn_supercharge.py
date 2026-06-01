# Author: Austin Seamann
# Last Update: 5-23-2025
# Version: 1.0 - Initial release, supports supercharging with PyRosetta threading

import json, time, os, sys, glob
import multiprocessing
import shutil
import warnings
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
import subprocess
import pickle

import argparse
import os
import pandas as pd

_pyrosetta_initialised = False

def _init_pyrosetta(step='this step'):
    """Lazily import and initialise PyRosetta. Exits with a clear message if not installed."""
    global _pyrosetta_initialised
    if _pyrosetta_initialised:
        return
    try:
        from pyrosetta import init
        init(silent=True)
        _pyrosetta_initialised = True
    except ImportError:
        sys.exit(
            f"""PyRosetta is required for {step} but is not installed.
            Install it from https://www.pyrosetta.org/downloads, then re-run.
            See the README for full setup instructions.\n"""
        )

VERBOSE = False
# Global ProteinMPNN methods
_scores = None
_S_to_seq = None
tied_featurize = None
parse_PDB = None
parse_fasta = None
StructureDatasetPDB = None
ProteinMPNN = None


def parse_catalytic_residues(catalytic_str):
    '''
    Parses the catalytic residues form a straing and returns a list of residues sequence numbers
    '''
    return [int(res.strip()) for res in catalytic_str.split(',')]

def net_charge(seq, add_histidine=False):
    '''
    Calculate the net charge of a protein sequence
    '''
    charge = seq.count('K') + seq.count('R') - seq.count('D') - seq.count('E')
    if add_histidine:
        charge += seq.count('H')
    return charge

def surface_selector(pose):
    '''
    Given a PDB, return the surface residues
    '''
    # Adapted from Joey Utils
    from pyrosetta.rosetta.core.select.residue_selector import LayerSelector

    # Init layer selector
    layer_selector = LayerSelector()

    # Checking surface
    layer_selector.set_layers(0, 0, 1)
    surface_selection = layer_selector.apply(pose)
    # Return PDB numbering
    surface_residues = [int(pose.pdb_info().pose2pdb(res).split()[0]) for res in range(1, pose.total_residue() + 1) if surface_selection[res]]
    return surface_residues

def index_selector(indices):
	"""
	Creates an index selector for a given selection. If a string is given, 
	uses that string directly in the selector. Converts integers to strings for 
	selector input.	Converts ranges or lists of integers to comma-separated 
	strings for selector input. - Adapted from Joey Utils
	"""
	from pyrosetta.rosetta.core.select.residue_selector import \
		ResidueIndexSelector

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
    from pyrosetta.rosetta.core.select.residue_selector import \
        InterGroupInterfaceByVectorSelector
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
    from pyrosetta import get_fa_scorefxn
    fr = FastRelax(repeats)
    if score_function is None:
        score_function = get_fa_scorefxn()
    fr.set_scorefxn(score_function)
    if task_factory:
        fr.set_task_factory(task_factory)
    if movemap:
        fr.set_movemap(movemap)
    return fr

def extract_pose_chain(pose, chain):
    '''
    Returns a pose containing only the specified chain (letter or int).
    Returns a copy if the pose is already single-chain. - Adapted from Joey Utils
    '''
    from pyrosetta import Pose
    if pose.num_chains() == 1:
        return Pose(pose)
    if isinstance(chain, int):
        chain_start = pose.chain_begin(chain)
        chain_end = pose.chain_end(chain)
    else:
        for c in range(1, pose.num_chains() + 1):
            chain_start = pose.chain_begin(c)
            chain_id_letter = pose.pdb_info().pose2pdb(chain_start).split()[1]
            if chain_id_letter == chain:
                chain_end = pose.chain_end(c)
                break
    return Pose(pose, chain_start, chain_end)

def total_energy(pose, score_function):
    '''
    Calculates total energy of a pose using TotalEnergyMetric. - Adapted from Joey Utils
    '''
    from pyrosetta.rosetta.core.simple_metrics.metrics import TotalEnergyMetric
    tem = TotalEnergyMetric()
    tem.set_scorefunction(score_function)
    return tem.calculate(pose)

def neighbor_selector(pose, catalytic_resi, distance):
    '''
    Given a PDB and catalytic residues, return the residues within a distance of the catalytic residues
    Adapted from Joey Utils
    '''
    global VERBOSE
    from pyrosetta.rosetta.core.select.residue_selector import NeighborhoodResidueSelector
    from pyrosetta.rosetta.core.simple_metrics.metrics import SelectedResiduesMetric

    # TODO: Make generic
    # Set selector on catalytic residues - using PDB numbering
    catalytic_resi = [pose.pdb_info().pdb2pose('A', res) for res in catalytic_resi]
    catalytic_selector = index_selector(catalytic_resi)

    if VERBOSE:
        srm = SelectedResiduesMetric()
        srm.set_residue_selector(catalytic_selector)
        srm.apply(pose)
        print("Catalytic residues:", catalytic_resi)

    # Setup neighbor selector
    if distance > 0.0:
        neighbor_selector = NeighborhoodResidueSelector()
        neighbor_selector.set_focus_selector(catalytic_selector)
        neighbor_selector.set_distance(distance)

        # Apply selector - and collect indices (PDB numbering)
        neighbor_selection = neighbor_selector.selection_positions(pose)
    
        positions_out = [int(pose.pdb_info().pose2pdb(res).split()[0]) for res in neighbor_selection]
    else:
        positions_out = catalytic_resi

    return positions_out

def strong_hbond_indices(pose, indices, no_fastrelax=False):
    '''
    Given a PDB and surface residue list, return the indices of the residues with strong hbonds and updated list with those removed
    '''
    from pyrosetta.rosetta.core.scoring.hbonds import HBondSet
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta import get_fa_scorefxn

    # Relax before calculating hbonds
    if not no_fastrelax:
        relax = FastRelax(5)
        relax.set_scorefxn(get_fa_scorefxn())
        relax.apply(pose)

    # Get hbond set
    hbond_set = HBondSet()
    pose.update_residue_neighbors()
    hbond_set.setup_for_residue_pair_energies(pose, False, False)

    hbond_indices = []
    for i in range(1, hbond_set.nhbonds() + 1):
        hbond = hbond_set.hbond(i)
        if hbond.energy() >= 0.5:  # semi-arbitrary cutoff - from original Rosetta Supercharge Protocol
            continue

        if hbond.don_res() in indices and not hbond.don_hatm_is_protein_backbone():
            hbond_indices.append(int(pose.pdb_info().pose2pdb(hbond.don_res()).split()[0]))
        if hbond.acc_res() in indices and not hbond.acc_atm_is_protein_backbone():
            hbond_indices.append(int(pose.pdb_info().pose2pdb(hbond.acc_res()).split()[0]))
    
    # Remove hbond indices from supercharge indices
    indices = [i for i in indices if i not in hbond_indices]

    return hbond_indices, indices

def charge_target(seq, target_charge, cur_indices, add_histidine=False):
    '''
    Remove residues from list if they are favorable for net charge - also, if net charge is not possible, report
    '''
    net_charge_init = net_charge(seq, add_histidine)
    delta_charge = target_charge - net_charge_init

    pos_list = ['K', 'R'] if not add_histidine else ['K', 'R', 'H']
    neg_list = ['D', 'E'] if not add_histidine else ['D', 'E']

    # Remove favorable charge residues from supercharge list
    if delta_charge > 0:
        indices = [i for i in cur_indices if seq[i-1] not in pos_list]  # need to have hist
        num_opposite = len([i for i in cur_indices if seq[i-1] in neg_list])
    elif delta_charge < 0:
        indices = [i for i in cur_indices if seq[i-1] not in neg_list]
        num_opposite = len([i for i in cur_indices if seq[i-1] in pos_list])
    else:
        print("Net charge already at target charge")
        return None
    
    charge_list = pos_list + neg_list
    num_other = len([i for i in indices if seq[i-1] not in charge_list])
    possible_charge_dist = num_other + (num_opposite * 2)

    if abs(delta_charge) > possible_charge_dist:
        print(f"Net charge impossible with target charge {target_charge}")
        return None
    
    return indices

def parse_for_supercharge(pdb, catalytic_str=None, distance=None, mutate_glyprocys=False, 
                          mutate_strong_hbond=False, no_fastrelax=False, add_histidine=False):
    '''
    Given a PDB, return the sequence and surface residues based on the target charge
    '''
    global VERBOSE
    _init_pyrosetta('surface residue detection')
    from pyrosetta import pose_from_pdb
    pose = pose_from_pdb(pdb)
    seq = pose.sequence()
    surface_residues = surface_selector(pose)
    indices = surface_residues  # hopefully this is just a list of indices -- 1-indexed

    if VERBOSE:
        print("Surface indices:", surface_residues)

    # Parse catalytic residues neighbors - remove from supercharge list
    if catalytic_str:
        if distance is None:
            raise ValueError("Must provide distance for catalytic residues - or set to 0.0")
        catalytic_residues = parse_catalytic_residues(catalytic_str)
        catalytic_residues_ext = neighbor_selector(pose, catalytic_residues, distance)
        indices = [i for i in indices if i not in catalytic_residues_ext]
        
        if VERBOSE:
            print("Catalytic indices:", catalytic_residues)
            print("Catalytic extended indices:", catalytic_residues_ext)
            print("Adjusted indices:", indices)
    
    # Parse strong hbond residues - remove from supercharge list
    if not mutate_strong_hbond:
        hbond_indices, indices = strong_hbond_indices(pose, indices, no_fastrelax)
        if VERBOSE:
            print("Strong hbond indices:", hbond_indices)
            print("Adjusted indices:", indices)
    
    # Remove glycine, proline, and cysteine from supercharge list - don't mutate
    if not mutate_glyprocys:
        indices = [i for i in indices if seq[i-1] not in ['G', 'P', 'C']]
        if VERBOSE:
            print("Glycine, proline, and cysteine indices:", [i for i in range(1, len(seq) + 1) if seq[i-1] in ['G', 'P', 'C']])
            print("Adjusted indices:", indices)

    # Determine initial net charge
    net_charge_init = net_charge(seq, add_histidine)

    return seq, indices, net_charge_init

def sample_sequence(model, X, S, mask, chain_M, chain_M_pos, residue_idx, chain_encoding_all,
                    initial_charge, target_charge, args, omit_AAs_np, bias_AAs_np,
                    omit_AA_mask, bias_by_idx_all, randn_1, verbose, pdb_id, max_temp_dict):
    global VERBOSE, _scores, _S_to_seq, tied_featurize, parse_PDB, parse_fasta, StructureDatasetPDB, ProteinMPNN
    count_sample = 0
    temp = args.temperature
    if VERBOSE:
        print("Current max_temp_dict:", max_temp_dict)

    # Check if target charge is in max_temp_dict - use as starting temp
    if target_charge in max_temp_dict:
        temp = max_temp_dict[target_charge]
        if VERBOSE:
            print(f"Using max temp for target charge {target_charge}: {temp}")

    while True:
        # Sample sequence
        sample_dict = model.supercharge_sample(X, initial_charge, target_charge, args.add_histidine, S, chain_M,
                                                chain_encoding_all, residue_idx, mask=mask, temperature=temp,
                                                prob_threshold=args.probability_threshold, omit_AAs_np=omit_AAs_np,
                                                bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos,
                                                omit_AA_mask=omit_AA_mask, bias_by_res=bias_by_idx_all,
                                                verbose=verbose)
        count_sample += 1

        if not args.unrestrict or sample_dict['charge'] == target_charge:
            # Save max temp for target charge
            if target_charge not in max_temp_dict:
                max_temp_dict[target_charge] = temp
            break

        if count_sample < 3:
            if temp <= 0.8:
                temp += 0.1
                temp = round(temp, 1)
                count_sample = 0
            elif temp > 0.9:
                max_temp_dict[target_charge] = temp
                print(f"Temperature too high for {pdb_id} with target charge {target_charge}")
                break
            print("Increasing temperature to: ", temp)

    S_sample = sample_dict["S"]
    log_probs = model(X, S_sample, mask, chain_M * chain_M_pos, residue_idx, chain_encoding_all, randn_1)
    mask_for_loss = mask * chain_M * chain_M_pos
    scores = _scores(S_sample, log_probs, mask_for_loss).cpu().data.numpy()
    global_scores = _scores(S_sample, log_probs, mask).cpu().data.numpy()
    seq = _S_to_seq(S_sample[0], chain_M[0])
    return scores.item(), global_scores.item(), seq, temp, max_temp_dict

def thread_sequence(pose, design_seq, og_seq, output_path, num_relax=5, verbose=None):
    '''
    Thread a designed sequence onto a backbone structure using PyRosetta FastRelax.
    Identifies mutations between og_seq and design_seq, then builds a constrained
    FastRelax that forces only the mutated residues and repacks the surrounding shell.
    The lowest-energy structure across num_relax runs is saved to output_path.
    verbose defaults to the global VERBOSE flag when not explicitly provided.
    '''
    global VERBOSE
    if verbose is None:
        verbose = VERBOSE
    _init_pyrosetta('sequence threading')
    from pyrosetta import Pose, get_fa_scorefxn
    sfxn = get_fa_scorefxn()

    # Identify mutated and native positions (1-indexed, stop at '/' delimiter)
    mutant_index_list = []
    mutation_list = []
    native_index_list = []
    for i, (wt_aa, des_aa) in enumerate(zip(og_seq, design_seq)):
        if des_aa == '/':
            break
        if des_aa != wt_aa:
            mutant_index_list.append(i + 1)
            mutation_list.append(des_aa)
        else:
            native_index_list.append(i + 1)

    if verbose:
        print(f"  Threading {len(mutant_index_list)} mutations onto {output_path}")
        print(f"  Mutations: {dict(zip(mutant_index_list, mutation_list))}")

    # If no mutations, just write the WT backbone
    if not mutant_index_list:
        pose.dump_pdb(output_path)
        return

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


def _thread_worker(task):
    '''
    Top-level worker for multiprocessing-based threading.
    Each worker spawns its own PyRosetta instance (required — PyRosetta cannot
    be shared across forked processes) and loads the backbone from disk.
    '''
    pdb_path, design_seq, og_seq, output_path, num_relax, verbose = task
    # Each spawned process must initialise PyRosetta independently
    _init_pyrosetta('sequence threading')
    from pyrosetta import pose_from_pdb
    pose = pose_from_pdb(pdb_path)
    thread_sequence(pose, design_seq, og_seq, output_path, num_relax, verbose)
    print(f'  Wrote {output_path}', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate scores for a PDB or set of PDBs with ProteinMPNN')
    # IO arguments
    parser.add_argument('-i', '--input', type=str, help="Input PDB or directory of PDBs", required=True)
    parser.add_argument('-o', '--output', type=str, help="Output Fasta file")
    parser.add_argument('-n', '--num_samples', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('--chain_id', type=str, default='A', help='Chain ID to design')
    # Preservation arguments
    parser.add_argument('-cat', '--catalytic', type=str, help='Comma-separated list of catalytic residues sequence numbers')
    parser.add_argument('-d', '--distance', type=float, default=None, help='Distance cutoff for non-mutable residues')
    parser.add_argument('-f', '--fixed_positions_jsonl', type=str, help='JSONL file with fixed positions')
    # Model arguments
    parser.add_argument('--model', type=str, default='v_48_020', help='Use alternative model checkpoints: v_48_002, v_48_010, v_48_020, v_48_030, ALL')
    parser.add_argument('--weights', type=str, default='original', help='Use alternative model weights: original, soluble [Needs to be non-pip version]')
    parser.add_argument('--path_to_weights', type=str, default='/home/als515/GitHub_Repos/ProteinMPNN', help='Path to weights file')
    # Supercharging arguments
    parser.add_argument('-c', '--target_charge', type=int, default=0, help='Target charge for supercharging')
    parser.add_argument('-top', '--top_charge', type=int, help='Top charge for supercharging - if scanning')
    parser.add_argument('-bottom', '--bottom_charge', type=int, help='Bottom charge for supercharging - if scanning')
    parser.add_argument('-p', '--probability_threshold', type=float, default=0.01, help='Probability threshold for sampling')
    parser.add_argument('-t', '--temperature', type=float, default=0.3, help='Temperature for sampling')
    parser.add_argument('-u', '--unrestrict', action='store_true', default=False, help='Unrestrict temp if charge not met')
    parser.add_argument('-gpc', '--mutate_glyprocys', action='store_true', default=False, help='Allow mutation of glycine, proline, or cysteine')
    parser.add_argument('-mhbond', '--mutate_hbonded_sidechains', action='store_true', default=False, help='Allow mutation of sidechains with strong hbonds')
    parser.add_argument('-nofast', '--no_fastrelax', action='store_true', default=False, help='Do not use FastRelax to remove strong hbonds - conditional on mutate_hbonded_sidechains')
    parser.add_argument('-addhis', '--add_histidine', action='store_true', default=False, help='Add histidine to the supercharge list (+1 charge)')
    # General arguments
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    # Threading arguments
    parser.add_argument('--thread', action='store_true', default=False,
                        help='Thread designed sequences onto the WT backbone and relax (creates threaded/ directory)')
    parser.add_argument('--thread_dir', type=str, default='threaded',
                        help='Output directory name for threaded structures (default: threaded)')
    parser.add_argument('--thread_workers', type=int, default=8,
                        help='Number of parallel workers for threading (default: 8). '
                             'Each worker spawns an independent PyRosetta process.')
    return parser.parse_args()

def main(args):
    global VERBOSE, _scores, _S_to_seq, tied_featurize, parse_PDB, parse_fasta, StructureDatasetPDB, ProteinMPNN
    if args.verbose:
        VERBOSE = True

    # Collect pdbs
    if os.path.isdir(args.input):
        pdbs = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith('.pdb')]
    else:
        print("Must provide a directory of PDBs")
        sys.exit()
    
    # Load model checkpoint
    # Load model
    if args.model == 'ALL':
        models = ['v_48_002', 'v_48_010', 'v_48_020', 'v_48_030']
    else:
        models = [args.model]
    if args.weights == 'original':
        weights_path = os.path.join(args.path_to_weights, 'vanilla_model_weights')
    elif args.weights == 'soluble':
        weights_path = os.path.join(args.path_to_weights, 'soluble_model_weights')
    checkpoint_paths = [os.path.join(weights_path, f'{model}.pt') for model in models]

    # Get path to protein_mpnn_utils from path_to_weights
    sys.path.append(args.path_to_weights)
    
    from protein_mpnn_utils import _scores, _S_to_seq, tied_featurize, parse_PDB, parse_fasta
    from protein_mpnn_utils import StructureDatasetPDB, ProteinMPNN

    # Pass to global scope - utils
    globals().update({
        '_scores': _scores,
        '_S_to_seq': _S_to_seq,
        'tied_featurize': tied_featurize,
        'parse_PDB': parse_PDB,
        'parse_fasta': parse_fasta,
        'StructureDatasetPDB': StructureDatasetPDB,
        'ProteinMPNN': ProteinMPNN
    })

    # Prepare ProteinMPNN - copied from protein_mpnn_run.py
    seed=int(np.random.randint(0, high=999, size=1, dtype=int)[0])

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)   
    
    hidden_dim = 128
    num_layers = 3 
    
    omit_AAs_list = 'X'
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    alphabet_dict = dict(zip(alphabet, range(21)))
    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.fixed_positions_jsonl:
        if os.path.isfile(args.fixed_positions_jsonl):
            with open(args.fixed_positions_jsonl, 'r') as json_file:
                json_list = list(json_file)
            for json_str in json_list:
                fixed_positions_dict = json.loads(json_str)
    else:
        fixed_positions_dict = None
    
    # Extend fixed_positions_dict to include non-surface/supercharged residues - not this is all currently 1-indexed
    initial_net_charge = {}  # Store the initial net charge of the protein (pdb: net_charge)
    chain_id = args.chain_id
    wt_seqs = {}
    wt_charge = {}

    # Save parsed sequences and indices to pickle
    os.makedirs('parsed', exist_ok=True)

    for pdb in pdbs:
        pdb_file = os.path.basename(pdb).split('.')[0]

        # ── Base cache: surface indices (never depends on axis params) ──────────
        pkl_file = f'parsed/{pdb_file}_seq_indices.pkl'
        if os.path.exists(pkl_file):
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            wt_seq, indices, net_charge_init = data[:3]
            if VERBOSE:
                print(f"Loaded surface cache: {pkl_file}", flush=True)
        else:
            wt_seq, indices, net_charge_init = parse_for_supercharge(pdb, args.catalytic,
                                        args.distance, args.mutate_glyprocys,
                                        args.mutate_hbonded_sidechains,
                                        args.no_fastrelax, args.add_histidine)
            with open(pkl_file, 'wb') as f:
                pickle.dump((wt_seq, indices, net_charge_init), f)
            with open(f'parsed/{pdb_file}_seq_indices.txt', 'w') as f:
                f.write(f'{wt_seq}\n')
                f.write(f'{indices}\n')
            if VERBOSE:
                print(f"Saved surface cache: {pkl_file}", flush=True)

        if fixed_positions_dict is None:
            fixed_positions_dict = {pdb_file: {chain_id:[]}}
        for i in range(1, len(wt_seq) + 1):
            if i not in indices:
                fixed_positions_dict[pdb_file][chain_id].append(i)
        initial_net_charge[pdb_file] = net_charge_init

        if VERBOSE:
            print("PDB:", pdb_file, flush=True)
            print("Fixed positions dict:", flush=True)
            print(fixed_positions_dict, flush=True)
            print("Initial net charge:", flush=True)
            print(initial_net_charge, flush=True)
            print("Initial seq", flush=True)
            print(wt_seq, flush=True)
        wt_seqs[pdb_file] = wt_seq
        wt_charge[pdb_file] = net_charge_init
        
    chain_id_dict = None
        
    # Unnecessary for my purposes
    pssm_dict = None
    omit_AA_dict = None
    tied_positions_dict = None
    bias_by_idx_dict = None
    bias_AAs_np = np.zeros(len(alphabet))

    for pdb in pdbs:
        pdb_id = os.path.basename(pdb).split('.')[0]
        # Load PDBs
        # Austin - might need to modify this to work with my PDB list already curated
        pdb_dict_list = parse_PDB(pdb, ca_only=False)
        dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=200000)
        all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9]=='seq_chain'] #['A','B', 'C',...]

        # Clamp fixed_positions_dict to the actual chain lengths seen by ProteinMPNN.
        # PyRosetta may count more residues than ProteinMPNN parses (e.g. non-standard
        # residues, chain-break terminators), causing out-of-bounds index errors in
        # tied_featurize if we don't filter here.
        if fixed_positions_dict and pdb_id in fixed_positions_dict:
            for ch in list(fixed_positions_dict[pdb_id].keys()):
                # Non-designed chains are fully fixed via chain_id_dict; skip clamping
                if ch != args.chain_id:
                    continue
                seq_key = f'seq_chain_{ch}'
                if seq_key in pdb_dict_list[0]:
                    chain_len = len(pdb_dict_list[0][seq_key])
                    before = len(fixed_positions_dict[pdb_id][ch])
                    fixed_positions_dict[pdb_id][ch] = [
                        p for p in fixed_positions_dict[pdb_id][ch] if p <= chain_len
                    ]
                    dropped = before - len(fixed_positions_dict[pdb_id][ch])
                    if dropped and VERBOSE:
                        print(f"  Clamped {dropped} out-of-bounds fixed position(s) for chain {ch} "
                              f"(ProteinMPNN chain length: {chain_len})", flush=True)
        designed_chain_list = [args.chain_id]
        fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
        chain_id_dict = {}
        chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)
        max_temp_dict = {}

        # My iteration
        for checkpoint_path in checkpoint_paths:
            checkpoint = torch.load(checkpoint_path, map_location=device) 
            model = ProteinMPNN(ca_only=False, num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, 
                                num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=0.00, k_neighbors=checkpoint['num_edges'])
            model.to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Validation epoch
            with torch.no_grad():
                score_list = []
                global_score_list = []
                seq_list = []
                temp_list = []
                for ix, protein in enumerate(dataset_valid):
                    batch_clones = [copy.deepcopy(protein) for i in range(1)]
                    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_idx_all, tied_beta = tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_idx_dict, ca_only=False)
                
                # Supercharging sampling
                # Sample based on target charge
                if not args.top_charge and not args.bottom_charge:
                    updated_indices = charge_target(wt_seq, args.target_charge, indices, args.add_histidine)
                    if updated_indices is None:
                        # Message printed in method
                        continue
                    else:
                        # Update mask - chain_M_pos
                        chain_M_pos = torch.zeros_like(chain_M)
                        for i in range(len(updated_indices)):
                            chain_M_pos[0, updated_indices[i]-1] = 1
                    if VERBOSE:
                        print("Final indices:", updated_indices)

                    for i in range(args.num_samples):
                        randn_1 = torch.randn(chain_M.shape, device=X.device)
                        log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
                        mask_for_loss = mask*chain_M*chain_M_pos
                        scores = _scores(S, log_probs, mask_for_loss) #score only the redesigned part
                        native_score = scores.cpu().data.numpy().item()
                        global_scores = _scores(S, log_probs, mask) #score the whole structure-sequence
                        global_native_score = global_scores.cpu().data.numpy().item()

                        # Run sample
                        score, global_score, seq, temp, max_temp_dict = sample_sequence(
                            model, X, S, mask, chain_M, chain_M_pos, residue_idx,
                            chain_encoding_all, initial_net_charge[pdb_id], args.target_charge, args, omit_AAs_np, bias_AAs_np,
                            omit_AA_mask, bias_by_idx_all, randn_1, VERBOSE, pdb_id, max_temp_dict
                        )

                        # Append to lists
                        score_list.append(score)
                        global_score_list.append(global_score)
                        seq_list.append(seq)
                        temp_list.append(temp)

                # Sample range of charges
                else:
                    for target_charge in range(args.bottom_charge, args.top_charge + 1):
                        updated_indices = charge_target(wt_seq, target_charge, indices, args.add_histidine)
                        if updated_indices is None:
                            # Message printed in method
                            continue
                        else:
                            # Update mask - chain_M_pos
                            chain_M_pos = torch.zeros_like(chain_M)
                            for i in range(len(updated_indices)):
                                chain_M_pos[0, updated_indices[i]-1] = 1
                        if VERBOSE:
                            print("Final indices:", updated_indices)
                        for i in range(args.num_samples):
                            randn_1 = torch.randn(chain_M.shape, device=X.device)
                            log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
                            mask_for_loss = mask*chain_M*chain_M_pos
                            scores = _scores(S, log_probs, mask_for_loss) #score only the redesigned part
                            native_score = scores.cpu().data.numpy().item()
                            global_scores = _scores(S, log_probs, mask) #score the whole structure-sequence
                            global_native_score = global_scores.cpu().data.numpy().item()
                            
                            # Run sample
                            score, global_score, seq, temp, max_temp_dict = sample_sequence(
                                model, X, S, mask, chain_M, chain_M_pos, residue_idx,
                                chain_encoding_all, initial_net_charge[pdb_id], target_charge, args, omit_AAs_np, bias_AAs_np,
                                omit_AA_mask, bias_by_idx_all, randn_1, VERBOSE, pdb_id, max_temp_dict
                            )

                            # Append to lists
                            score_list.append(score)
                            global_score_list.append(global_score)
                            seq_list.append(seq)
                            temp_list.append(temp)
        # --- Per-sequence detailed output (verbose only) ---
        if VERBOSE:
            print(f"\nPDB: {pdb_id}", flush=True)
            print(f"  WT   charge={wt_charge[pdb_file]:+d}  score={native_score:.4f}  global_score={global_native_score:.4f}")
            print(f"       {wt_seqs[pdb_file]}")
            for i, seq in enumerate(seq_list):
                print(f"  [{i:>2}] charge={net_charge(seq, args.add_histidine):+d}  score={score_list[i]:.4f}  "
                      f"global_score={global_score_list[i]:.4f}  temp={temp_list[i]:.1f}")
                print(f"       {seq}")

        # --- Summary table (always printed) ---
        print(f"\n{'='*72}")
        print(f"  Results: {pdb_id}  ({len(seq_list)} sample(s))")
        print(f"{'='*72}")
        print(f"  {'ID':<5}  {'Charge':>7}  {'Score':>10}  {'Global Score':>13}  {'Temp':>5}")
        print(f"  {'-'*60}")
        print(f"  {'WT':<5}  {wt_charge[pdb_file]:>+7d}  {native_score:>10.4f}  {global_native_score:>13.4f}  {'N/A':>5}")
        for i, seq in enumerate(seq_list):
            print(f"  {i:<5}  {net_charge(seq, args.add_histidine):>+7d}  {score_list[i]:>10.4f}  "
                  f"{global_score_list[i]:>13.4f}  {temp_list[i]:>5.1f}")
        print(f"{'='*72}", flush=True)

        # Write to fasta
        for pdb in pdbs:
            if args.output:
                output_file = args.output
            else:
                output_file = f'{pdb_file}_sc_{args.target_charge}.fasta'
            pdb_file = os.path.basename(pdb).split('.')[0]
            with open(output_file, 'w') as f:
                # Write native
                native_seq = wt_seqs[pdb_file]
                native_charge = wt_charge[pdb_file]
                f.write(f'>{pdb_file},charge={native_charge},score={native_score:.4f},global_score={global_native_score:.4f}\n')
                f.write(f'{native_seq}\n')
                # Write sampled
                for i, seq in enumerate(seq_list):
                    f.write(f'>{pdb_id}_{i},charge={net_charge(seq, args.add_histidine)},score={score_list[i]:.4f},global_score={global_score_list[i]:.4f},temperature={temp_list[i]}\n')
                    f.write(f'{seq}\n')

        # --- Thread designed sequences onto backbone if requested ---
        if args.thread:
            os.makedirs(args.thread_dir, exist_ok=True)
            native_seq = wt_seqs[pdb_id]
            tasks = []
            label_muts = args.target_charge
            for idx, seq in enumerate(seq_list):
                out_name = f'{args.thread_dir}/{pdb_id}_{label_muts}_{idx}.pdb'
                tasks.append((pdb, seq, native_seq, out_name, 5, VERBOSE))
            num_workers = min(args.thread_workers, len(tasks))
            print(f"\nThreading {len(tasks)} sequence(s) for {pdb_id} "
                  f"across {num_workers} worker(s)...", flush=True)
            if num_workers > 1:
                # Use 'spawn' — PyRosetta cannot be safely shared across forked
                # processes; each spawned worker initialises its own instance.
                ctx = multiprocessing.get_context('spawn')
                with ctx.Pool(processes=num_workers) as pool:
                    pool.map(_thread_worker, tasks)
            else:
                # Single-worker path: reuse the already-initialised PyRosetta instance
                from pyrosetta import pose_from_pdb
                thread_pose = pose_from_pdb(pdb)
                for task in tasks:
                    _, seq, og_seq, out_name, num_relax, verbose = task
                    thread_sequence(thread_pose, seq, og_seq, out_name, num_relax, verbose)
                    print(f'  Wrote {out_name}', flush=True)

if __name__ == '__main__':
    args = parse_args()
    main(args)
