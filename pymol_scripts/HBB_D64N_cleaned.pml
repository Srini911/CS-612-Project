load proteins_cleaned/HBB_D64N_cleaned_docked.pdb, protein
load proteins_cleaned/hydroxyurea.pdb, ligand

hide everything
show cartoon, protein
color cyan, protein

show sticks, ligand
color yellow, ligand

zoom ligand
orient

mset 1 x60
util.mroll(1,60)

mpng proteins_cleaned/frames/HBB_D64N_cleaned_frame
quit
