import argparse
import os
from Bio.PDB import PDBParser, NeighborSearch
import pyrosetta

def parse_catalytic_residues(catalytic_str):
    # Parses the catalytic residues from a string and returns a list of residue sequence numbers
    return [int(res.strip()) for res in catalytic_str.split(',')]

def collect_interface(pdb_filename, catalytic_residues, distance):
    # Parse the PDB file using Biopython
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_filename)
    
    # Collect all atoms and calculate Pyrosetta-compatible numbering
    all_atoms = [atom for atom in structure.get_atoms()]
    residues = list(structure.get_residues())
    residue_map = {res.get_id()[1]: i + 1 for i, res in enumerate(residues)}  # Map Biopython numbering to Pyrosetta numbering
    
    # Translate catalytic residue numbers to Pyrosetta numbering
    catalytic_pyrosetta = [residue_map[res] for res in catalytic_residues if res in residue_map]
    if not catalytic_pyrosetta:
        raise ValueError("None of the specified catalytic residues were found in the PDB file.")
    
    # Print catalytic residues in Pyrosetta numbering with their residue types
    print("Catalytic residues (PDB numbering -> Pyrosetta numbering):")
    for res in catalytic_residues:
        if res in residue_map:
            pyros_res = residue_map[res]
            residue = residues[pyros_res - 1]  # Get the residue from Pyrosetta numbering
            print(f"PDB: {res}, Pyrosetta: {pyros_res}, Type: {residue.get_resname()}")
    
    # Identify catalytic residue atoms
    catalytic_atoms = [
        atom for atom in all_atoms 
        if atom.get_parent().get_id()[1] in catalytic_residues
    ]
    
    if not catalytic_atoms:
        raise ValueError("No catalytic residues found in the PDB file.")
    
    # Use NeighborSearch to find residues within the distance cutoff
    ns = NeighborSearch(all_atoms)
    neighbor_residues = set()
    
    for atom in catalytic_atoms:
        close_atoms = ns.search(atom.coord, distance)
        for close_atom in close_atoms:
            parent_residue = close_atom.get_parent()
            res_id = parent_residue.get_id()[1]  # Sequence number
            chain_id = parent_residue.get_full_id()[2]  # Chain ID
            neighbor_residues.add((chain_id, residue_map[res_id]))  # Use Pyrosetta numbering
    
    # Return the sorted list of residue IDs
    return sorted(neighbor_residues)

def write_resfile(conserved_residues, output_file='resfile.txt'):
    # Write the RESfile to specify mutation constraints
    with open(output_file, 'w') as resfile:
        resfile.write('ALLAA\nstart\n')
        for chain_id, res_num in conserved_residues:
            resfile.write(f"{res_num} {chain_id or ' '} NATAA\n")
    return output_file

def super_charge_pdb(pose, charge, avnapsa=False):
    # Import and configure the Supercharge mover
    from pyrosetta.rosetta.protocols.design_opt import Supercharge
    supercharge_mover = Supercharge()

    # Set options directly on the Supercharge mover
    supercharge_mover.target_net_charge(charge)
    supercharge_mover.target_net_charge_active(True)

    if avnapsa:
        # AvNAPSA mode: deterministic, sequence-based surface definition
        if charge >= 0:
            supercharge_mover.AvNAPSA_positive(True)
        else:
            supercharge_mover.AvNAPSA_negative(True)
    else:
        # Rosetta score-based mode
        supercharge_mover.surface_residue_cutoff(16)
        supercharge_mover.pre_packminpack(True)
        supercharge_mover.dont_mutate_glyprocys(True)
        supercharge_mover.dont_mutate_correct_charge(True)
        supercharge_mover.dont_mutate_hbonded_sidechains(True)
        supercharge_mover.compare_residue_energies_all(False)
        supercharge_mover.compare_residue_energies_mut(True)

        # Flags specific for charge
        if charge >= 0:
            # Positive supercharge
            supercharge_mover.include_arg(True)
            supercharge_mover.include_lys(True)
            supercharge_mover.refweight_arg(-1.98)
            supercharge_mover.refweight_lys(-1.65)
        else:
            # Negative supercharge
            supercharge_mover.include_asp(True)
            supercharge_mover.include_glu(True)
            supercharge_mover.refweight_asp(-0.6)
            supercharge_mover.refweight_glu(-0.8)

    # Apply the Supercharge mover
    supercharge_mover.apply(pose)
    print("Net charge: ", supercharge_mover.get_net_charge(pose))

def parse_args():
    parser = argparse.ArgumentParser(description='Supercharge a protein structure')
    parser.add_argument('-input', type=str, help='Input PDB file', required=True)
    parser.add_argument('-top', type=int, help='Top bound for charge')
    parser.add_argument('-bottom', type=int, help='Bottom bound for charge')
    parser.add_argument('-catalytic', type=str, help='Comma-separated list of catalytic residue sequence numbers (PDB numbering)')
    parser.add_argument('-distance', type=float, default=8.0, help='Distance cutoff for identifying interface residues (default: 8.0 Å)')
    parser.add_argument('-charge', type=int, help='Target net charge for supercharging')
    parser.add_argument('-avnapsa', '--avnapsa', action='store_true', default=False, help='Use AvNAPSA protocol for supercharging instead of structure-based scoring')
    return parser.parse_args()

def main(args):
    # Collect interface residues using Biopython (if catalytic residues are provided)
    if args.catalytic:
        catalytic_residues = parse_catalytic_residues(args.catalytic)
        non_mutatable_residues = collect_interface(args.input, catalytic_residues, args.distance)
        res_file = write_resfile(non_mutatable_residues)
        pyrosetta.init(f"-resfile {res_file}")
    else:
        pyrosetta.init()
    
    pose = pyrosetta.pose_from_pdb(args.input)
    
    # Determine charges to iterate over
    if args.charge is not None:
        charges = [args.charge]
    else:
        charges = range(args.top, args.bottom + 1)
    
    # Supercharge the pdb
    for charge in charges:
        super_charge_pdb(pose.clone(), charge, avnapsa=args.avnapsa)

if __name__ == '__main__':
    args = parse_args()
    main(args)
