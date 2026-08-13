import argparse
import os
from pyrosetta import *
init('-ex1 -ex2')


def total_energy(pose, score_function, selection=None):
	"""
	Credit: Joey Lubin
	Calculates total energy of a pose using a TotalEnergyMetric. If a selector
	is provided, calculates the total energy of the selection rather than the
	whole pose.
	"""
	from pyrosetta.rosetta.core.simple_metrics.metrics import TotalEnergyMetric

	# Create the metric
	tem = TotalEnergyMetric()
	tem.set_scorefunction(score_function)
	
	# Add the selector
	if selection:
		tem.set_residue_selector(selection)
		
	return tem.calculate(pose)


def get_sf(rep_type='hard', symmetry=False, membrane=0, constrain=1.0, hbnet=0):
	"""
	Credit: Joey Lubin
	Determines the appropriate score function to use, based on a rep_type
	that is either hard (ref2015) or soft (ref2015_soft), whether symmetry 
	and/or membrane modeling are in use, and whether constraints are desired.
	If setting membrane and/or hbnet, change value to desired nonzero weight.
	"""
	from pyrosetta import create_score_function, ScoreFunction
	from pyrosetta.rosetta.core.scoring import ScoreType
	from pyrosetta.rosetta.core.scoring.symmetry import SymmetricScoreFunction

	score_types = {'hard': 'ref2015', 'soft': 'ref2015_soft'}
	assert rep_type in score_types

	# Create base empty score function symmetrically or asymmetrically
	if symmetry: # Declare symmetric score functions
		score_function = SymmetricScoreFunction()
	else:
		score_function = ScoreFunction()

	# Add main score weights
	if rep_type == 'hard':
		score_function.add_weights_from_file('ref2015')
	elif rep_type == 'soft':
		score_function.add_weights_from_file('ref2015_soft')
		if membrane: # Set up a soft-rep version of franklin2019 manually
			score_function.set_weight(ScoreType.fa_water_to_bilayer, membrane)

	# Add membrane weights if appliccable
	if membrane:
		score_function.add_weights_from_file('franklin2019')

	# The score functions do not have constraint weights incorporated in 
	# themselves. If requisite, the constraint weights are added.
	if constrain:
		score_function.set_weight(ScoreType.atom_pair_constraint, constrain)
		score_function.set_weight(ScoreType.coordinate_constraint, constrain)
		score_function.set_weight(ScoreType.angle_constraint, constrain)
		score_function.set_weight(ScoreType.dihedral_constraint, constrain)
		score_function.set_weight(ScoreType.metalbinding_constraint, constrain)
		score_function.set_weight(ScoreType.chainbreak, constrain)
		score_function.set_weight(ScoreType.res_type_constraint, constrain)

	# Optionally adding in hbnet
	if hbnet:
		score_function.set_weight(ScoreType.hbnet, hbnet)
	
	return score_function


def fast_relax_mover(score_function=None, task_factory=None, movemap=None, repeats=5):
	"""
	Credit: Joey Lubin
	Creates a FastRelax mover. If no score function is given, a default 
	ref15_cst will be used. If a task factory and/or movemap are provided, 
	they will also be incorporated into the mover. By default, FastRelax 
	goes through five ramping cycles, but this number can be adjusted with 
	the repeats option.
	"""
	from pyrosetta.rosetta.protocols.relax import FastRelax

	# Make FastRelax mover with given score function
	fr = FastRelax(repeats)

	# Set score function
	if score_function == None:
		score_function = get_sf()
	fr.set_scorefxn(score_function)

	# Set task factory
	if task_factory:
		fr.set_task_factory(task_factory)

	# Set move map
	if movemap:
		fr.set_movemap(movemap)

	return fr


def relax_pdb(pdb_in, pdb_out, repeats=5):
    """
    Relaxes a PDB file using the FastRelax mover. The relaxed structure is 
    written to a new PDB file.
    """
	# Setup score function
    sfxn = get_sf()
    best_energy = float('inf')
    top_pose = Pose()
    pose = pose_from_pdb(pdb_in)
	# Output location (remove extention and add _relaxed.pdb)
    pdb_id = os.path.basename(pdb_in).split('.')[0]
    pdb_out_tmp = os.path.join(pdb_out, pdb_id + '_relaxed.pdb')
    print(pdb_out_tmp)
    for i in range(repeats):
        top_pose.detached_copy(pose)
        relax_mover = fast_relax_mover(score_function=sfxn)
        relax_mover.apply(top_pose)
        # Collect energy of relaxed pose
        energy = total_energy(top_pose, sfxn)
		# Check if energy is better
        if energy < best_energy:
            best_energy = energy
			# Save the best pose
            top_pose.dump_pdb(pdb_out_tmp)
            print(pdb_out_tmp)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input file or path", type=str)
    parser.add_argument("-o", "--output", help="Output file or path of relaxed structures", type=str, default=".")
    parser.add_argument("-n", "--nrelax", help="Number of relaxations", type=int, default=5)
    parser.add_argument("-c", "--cpu_count", help="Number of CPUs to utilize - Default all", type=int, default=-1)
    args = parser.parse_args()
    return args


def main(args):
	# Setup output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
	# Determine if the input is a PDB or a directory
    if os.path.isdir(args.input):
		
        # Run with multiprocessing
        import multiprocessing as mp

        pdbs = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith('.pdb') or f.endswith('.pdb.gz') or f.endswith('.cif')]
        if args.cpu_count != -1:
            pool = mp.Pool(args.cpu_count)
        else:
            pool = mp.Pool(mp.cpu_count())
        pool.starmap(relax_pdb, [(pdb, args.output, args.nrelax) for pdb in pdbs])
        pool.close()
        pool.join()
    else:
        relax_pdb(args.input, args.output, args.nrelax)


if __name__ == "__main__":
    args = parse_args()
    main(args)
