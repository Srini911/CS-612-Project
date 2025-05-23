load ../proteins_cleaned/HBB_D64N_cleaned_docked.pdb, protein
load ../proteins_cleaned/hydroxyurea.pdb, ligand

hide everything

# Visual style: surface + transparent cartoon
show surface, protein
color gray70, protein
set transparency, 0.5, surface

show cartoon, protein
set cartoon_transparency, 0.7

show sticks, ligand
set stick_radius, 0.2
color yellow, ligand

zoom ligand, 10
select contacts, (protein within 4 of ligand)
show sticks, contacts
color red, contacts

mset 1 x60
util.mroll(1,60)

mpng ../proteins_cleaned/frames/HBB_D64N_interaction
quit
