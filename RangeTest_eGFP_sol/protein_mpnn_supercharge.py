# Author: Austin Seamann
# Last Update: 5-5-2025

import json, time, os, sys, glob
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
from pyrosetta import *
init(silent=True)

import argparse
import os
import pandas as pd

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

def parse_for_supercharge(pdb, catalytic_str=None, distance=None, mutate_glyprocys=False, mutate_strong_hbond=False, no_fastrelax=False, add_histidine=False):
    '''
    Given a PDB, return the sequence and surface residues based on the target charge
    '''
    global VERBOSE
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
                    omit_AA_mask, bias_by_res_all, randn_1, verbose, pdb_id, max_temp_dict):
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
                                                omit_AA_mask=omit_AA_mask, bias_by_res=bias_by_res_all,
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

def parse_args():
    parser = argparse.ArgumentParser(description='Generate scores for a PDB or set of PDBs with ProteinMPNN')
    parser.add_argument('-i', '--input', type=str, help="Input PDB or directory of PDBs", required=True)
    parser.add_argument('-o', '--output', type=str, help="Output Fasta file")
    parser.add_argument('-cat', '--catalytic', type=str, help='Comma-separated list of catalytic residues sequence numbers')
    parser.add_argument('-d', '--distance', type=float, default=None, help='Distance cutoff for non-mutable residues')
    parser.add_argument('--model', type=str, default='v_48_020', help='Use alternative model checkpoints: v_48_002, v_48_010, v_48_020, v_48_030, ALL')
    parser.add_argument('--weights', type=str, default='original', help='Use alternative model weights: original, soluble [Needs to be non-pip version]')
    parser.add_argument('--path_to_weights', type=str, default='/home/als515/GitHub_Repos/ProteinMPNN', help='Path to weights file')
    parser.add_argument('-f', '--fixed_positions_jsonl', type=str, help='JSONL file with fixed positions')
    parser.add_argument('-c', '--target_charge', type=int, default=0, help='Target charge for supercharging')
    parser.add_argument('-top', '--top_charge', type=int, help='Top charge for supercharging - if scanning')
    parser.add_argument('-bottom', '--bottom_charge', type=int, help='Bottom charge for supercharging - if scanning')
    parser.add_argument('-p', '--probability_threshold', type=float, default=0.01, help='Probability threshold for sampling')
    parser.add_argument('-t', '--temperature', type=float, default=0.3, help='Temperature for sampling')
    parser.add_argument('-u', '--unrestrict', action='store_true', default=False, help='Unrestrict temp if charge not met')
    parser.add_argument('-n', '--num_samples', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-mhbond', '--mutate_hbonded_sidechains', action='store_true', default=False, help='Allow mutation of sidechains with strong hbonds')
    parser.add_argument('-gpc', '--mutate_glyprocys', action='store_true', default=False, help='Allow mutation of glycine, proline, or cysteine')
    parser.add_argument('-nofast', '--no_fastrelax', action='store_true', default=False, help='Do not use FastRelax to remove strong hbonds')
    parser.add_argument('-addhis', '--add_histidine', action='store_true', default=False, help='Add histidine to the supercharge list (+1 charge)')
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
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

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
    chain_id = 'A'
    wt_seqs = {}
    wt_charge = {}

    # Save parsed sequences and indices to pickle
    os.makedirs('parsed', exist_ok=True)

    for pdb in pdbs:
        pdb_file = os.path.basename(pdb).split('.')[0]
        pkl_file = f'parsed/{pdb_file}_seq_indices.pkl'
        if os.path.exists(pkl_file):
            with open(pkl_file, 'rb') as f:
                wt_seq, indices, net_charge_init = pickle.load(f)
            if fixed_positions_dict is None:
                fixed_positions_dict = {pdb_file: {chain_id:[]}}
                for i in range(1, len(wt_seq) + 1):
                    if i not in indices:
                        fixed_positions_dict[pdb_file][chain_id].append(i)
            initial_net_charge[pdb_file] = net_charge_init
        else:
            wt_seq, indices, net_charge_init = parse_for_supercharge(pdb, args.catalytic, 
                                                                     args.distance, args.mutate_glyprocys,
                                                                     args.mutate_hbonded_sidechains, 
                                                                     args.no_fastrelax, args.add_histidine)
            if fixed_positions_dict is None:
                fixed_positions_dict = {pdb_file: {chain_id:[]}}
                for i in range(1, len(wt_seq) + 1):
                    if i not in indices:
                        fixed_positions_dict[pdb_file][chain_id].append(i)
            initial_net_charge[pdb_file] = net_charge_init
            
            # Save to pickle
            with open(pkl_file, 'wb') as f:
                pickle.dump((wt_seq, indices, net_charge_init), f)
            # Write as txt file
            with open(f'parsed/{pdb_file}_seq_indices.txt', 'w') as f:
                f.write(f'{wt_seq}\n')
                f.write(f'{indices}\n')
        
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
    bias_by_res_dict = None
    bias_AAs_np = np.zeros(len(alphabet))

    for pdb in pdbs:
        pdb_id = os.path.basename(pdb).split('.')[0]
        # Load PDBs
        # Austin - might need to modify this to work with my PDB list already curated
        pdb_dict_list = parse_PDB(pdb, ca_only=False)
        dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=200000)
        all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9]=='seq_chain'] #['A','B', 'C',...]
        designed_chain_list = all_chain_list
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
                    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict, ca_only=False)
                
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
                            omit_AA_mask, bias_by_res_all, randn_1, VERBOSE, pdb_id, max_temp_dict
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
                                omit_AA_mask, bias_by_res_all, randn_1, VERBOSE, pdb_id, max_temp_dict
                            )

                            # Append to lists
                            score_list.append(score)
                            global_score_list.append(global_score)
                            seq_list.append(seq)
                            temp_list.append(temp)

        # Print output
        if VERBOSE:
            print("PDB:", pdb_id, flush=True)
            print("Seq:", seq_list, flush=True)
            print("Score:", score_list, flush=True)
            print("Global Score:", global_score_list, flush=True)
            print("Temp:", temp_list, flush=True)

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

if __name__ == '__main__':
    args = parse_args()
    main(args)
