import os
import csv
import subprocess
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

# ─── BASE DIRECTORY SETUP ─────────────────────────────────────────────────────
base_dir = r"C:\Users\srini\Desktop\docking_project"
proteins_dir = os.path.join(base_dir, "proteins_cleaned")
ligand_path = os.path.join(base_dir, "ligand", "hydroxyurea.pdbqt")
csv_input = os.path.join(base_dir, "HBB_50Mutations_Data.csv")
csv_output = os.path.join(base_dir, "CS612 Dataset.csv")
ref_path = os.path.join(proteins_dir, "HBB_WT_cleaned.pdb")

# ─── STEP 1: Prepare Receptors ────────────────────────────────────────────────
def prepare_receptors():
    mgltools_python = r"C:\Program Files (x86)\MGLTools-1.5.7\python.exe"
    prepare_script = r"C:\Program Files (x86)\MGLTools-1.5.7\MGLToolsPckgs\AutoDockTools\Utilities24\prepare_receptor4.py"
    for file in os.listdir(proteins_dir):
        if file.endswith("_cleaned.pdb"):
            input_path = os.path.join(proteins_dir, file)
            output_path = os.path.join(proteins_dir, file.replace("_cleaned.pdb", ".pdbqt"))
            cmd = f'"{mgltools_python}" "{prepare_script}" -r "{input_path}" -o "{output_path}" -A hydrogens'
            subprocess.call(cmd, shell=True)

# ─── STEP 2: Generate Config Files ────────────────────────────────────────────
def generate_configs():
    config_dir = os.path.join(base_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)
    for file in os.listdir(proteins_dir):
        if file.endswith(".pdbqt"):
            base = file.replace(".pdbqt", "")
            with open(os.path.join(config_dir, f"{base}.txt"), "w") as f:
                f.write(f"""receptor = proteins_cleaned/{file}
ligand = ligand/hydroxyurea.pdbqt
center_x = 10
center_y = 10
center_z = 10
size_x = 20
size_y = 20
size_z = 20
out = results/{base}_out.pdbqt
log = logs/{base}_log.txt
""")

# ─── STEP 3: Run Docking ──────────────────────────────────────────────────────
def run_docking():
    config_dir = os.path.join(base_dir, "configs")
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
    for config_file in os.listdir(config_dir):
        if config_file.endswith(".txt"):
            subprocess.run(["vina", "--config", os.path.join(config_dir, config_file)], capture_output=True, text=True)

# ─── STEP 4: Extract Affinities ───────────────────────────────────────────────
def extract_affinities():
    output_csv = os.path.join(base_dir, "binding_affinities.csv")
    with open(output_csv, mode='w', newline='') as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(["Mutation", "Binding_Affinity"])
        for file in os.listdir(proteins_dir):
            if file.endswith("_docked.pdbqt"):
                with open(os.path.join(proteins_dir, file)) as f:
                    for line in f:
                        if line.startswith("REMARK VINA RESULT"):
                            score = float(line.split()[3])
                            writer.writerow([file.replace("_docked.pdbqt", ""), score])
                            break

# ─── STEP 5: Check Binding at Mutation ────────────────────────────────────────
def check_binding_proximity():
    output_csv = os.path.join(proteins_dir, "mutation_binding_check.csv")
    parser = PDBParser(QUIET=True)
    results = []
    for file in os.listdir(proteins_dir):
        if file.endswith("_cleaned_docked.pdbqt"):
            base = file.replace("_cleaned_docked.pdbqt", "")
            pdb_file = os.path.join(proteins_dir, base + "_cleaned.pdb")
            if not os.path.exists(pdb_file):
                continue
            with open(os.path.join(proteins_dir, file), 'r') as f:
                ligand_atoms = [np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                                for line in f if line.startswith("ATOM") or line.startswith("HETATM")]
            structure = parser.get_structure("protein", pdb_file)
            res_num = int(''.join(filter(str.isdigit, base)))
            res_atoms = []
            for residue in structure.get_residues():
                if residue.get_id()[1] == res_num:
                    res_atoms.extend([atom.coord for atom in residue])
            if res_atoms and ligand_atoms:
                min_dist = min(np.linalg.norm(r - l) for r in res_atoms for l in ligand_atoms)
                label = "Yes" if min_dist <= 5.0 else "No"
                results.append((file, res_num, round(min_dist, 2), label))
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["File", "MutationResidue", "MinDistance(Å)", "BindingAtMutationSite"])
        writer.writerows(results)

# ─── STEP 6: Merge Dataset ────────────────────────────────────────────────────
def merge_dataset():
    features_df = pd.read_csv(csv_input)
    binding_df = pd.read_csv(os.path.join(base_dir, "binding_affinities.csv"))
    features_df["Mutation"] = features_df["Mutation"].astype(str).str.strip().str.upper()
    binding_df["Mutation"] = binding_df["Mutation"].astype(str).str.strip().str.upper()
    df = pd.merge(features_df, binding_df, on="Mutation")
    df.to_csv(csv_output, index=False)

# ─── STEP 7: Add RMSD ─────────────────────────────────────────────────────────
def add_rmsd():
    parser = PDBParser(QUIET=True)
    ref_coords = [atom.get_coord() for atom in parser.get_structure("ref", ref_path).get_atoms() if atom.get_id() == 'CA']
    df = pd.read_csv(csv_output)
    def rmsd_calc(row):
        try:
            pdb_file = row['File'].replace("_docked.pdbqt", ".pdb")
            file_path = os.path.join(proteins_dir, pdb_file)
            coords = [atom.get_coord() for atom in parser.get_structure("x", file_path).get_atoms() if atom.get_id() == 'CA']
            if len(coords) != len(ref_coords):
                return None
            return round(np.sqrt(((np.array(coords) - ref_coords) ** 2).sum() / len(coords)), 3)
        except:
            return None
    df["RMSD"] = df.apply(rmsd_calc, axis=1)
    df.to_csv(csv_output, index=False)

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    prepare_receptors()
    generate_configs()
    run_docking()
    extract_affinities()
    check_binding_proximity()
    merge_dataset()
    add_rmsd()
    print("✔️ All steps completed. Final dataset saved at:", csv_output)
