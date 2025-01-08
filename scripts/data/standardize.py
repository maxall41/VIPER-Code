import glob
import hashlib
import os
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import fire
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from tqdm import tqdm

tqdm.pandas()


def generate_xyz_filename(smiles):
    """Generate MD5 hash-based filename for the given SMILES string."""
    return hashlib.md5(smiles.encode()).hexdigest() + ".xyz"


def mol_to_xyz_string(mol):
    """Convert RDKit mol object to XYZ file format string."""
    conf = mol.GetConformer()
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coords = conf.GetPositions()

    xyz_lines = [
        str(mol.GetNumAtoms()),
        "",
    ]  # Number of atoms and blank comment line
    for atom, (x, y, z) in zip(atoms, coords):
        xyz_lines.append(f"{atom:<2} {x:>10.4f} {y:>10.4f} {z:>10.4f}")

    return "\n".join(xyz_lines)


def save_mol_as_xyz(mol, filename):
    """Save molecule as XYZ file."""
    with open(filename, "w") as f:
        f.write(mol_to_xyz_string(mol))


def load_xyz_files(directory):
    xyz_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith(".xyz"):
            file_key = filename.rsplit(".")[0]
            file_path = os.path.join(directory, filename)

            atomic_symbols = []
            xyz_coordinates = []

            with open(file_path, "r") as file:
                for line_number, line in enumerate(file):
                    if line_number == 0:
                        num_atoms = int(line)
                    elif line_number == 1:
                        comment = line
                    else:
                        atomic_symbol, x, y, z = line.split()
                        atomic_symbols.append(atomic_symbol)
                        xyz_coordinates.append([float(x), float(y), float(z)])

            xyz_dict[file_key.split("_")[0]] = (atomic_symbols, xyz_coordinates)

    return xyz_dict


def standardize_smiles(smiles):
    # Compose the XML request
    xml_request = f"""
    <PCT-Data>
        <PCT-Data_input>
            <PCT-InputData>
                <PCT-InputData_standardize>
                    <PCT-Standardize>
                        <PCT-Standardize_structure>
                            <PCT-Structure>
                                <PCT-Structure_structure>
                                    <PCT-Structure_structure_string>{smiles}</PCT-Structure_structure_string>
                                </PCT-Structure_structure>
                                <PCT-Structure_format>
                                    <PCT-StructureFormat value="smiles"/>
                                </PCT-Structure_format>
                            </PCT-Structure>
                        </PCT-Standardize_structure>
                        <PCT-Standardize_oformat>
                            <PCT-StructureFormat value="smiles"/>
                        </PCT-Standardize_oformat>
                    </PCT-Standardize>
                </PCT-InputData_standardize>
            </PCT-InputData>
        </PCT-Data_input>
    </PCT-Data>
    """

    # Send the request to PubChem
    response = requests.post(
        "https://pubchem.ncbi.nlm.nih.gov/pug/pug.cgi", data=xml_request
    )

    # Parse the response XML
    root = ET.fromstring(response.text)

    status_root = None
    # Check if the request is still running
    if root.find(".//PCT-OutputData_output_waiting") is not None:
        # Extract the request ID
        request_id = root.find(".//PCT-Waiting_reqid").text
        runs = 0
        while True:
            # Compose the status check XML
            status_xml = f"""
            <PCT-Data>
                <PCT-Data_input>
                    <PCT-InputData>
                        <PCT-InputData_request>
                            <PCT-Request>
                                <PCT-Request_reqid>{request_id}</PCT-Request_reqid>
                                <PCT-Request_type value="status"/>
                            </PCT-Request>
                        </PCT-InputData_request>
                    </PCT-InputData>
                </PCT-Data_input>
            </PCT-Data>
            """

            # Send the status check request
            status_response = requests.post(
                "https://pubchem.ncbi.nlm.nih.gov/pug/pug.cgi", data=status_xml
            )
            status_root = ET.fromstring(status_response.text)

            # Check if the request is completed
            if (
                status_root.find(".//PCT-OutputData_output_structure")
                is not None
            ):
                break

            # Wait for a short interval before checking again
            time.sleep(1)
            runs += 1
            if runs > 10:
                print("FAILED - Time Out")
                return None

    # Extract the standardized SMILES
    standardized_smiles = status_root.find(
        ".//PCT-Structure_structure_string"
    ).text

    return standardized_smiles


def process_molecules(df, xyz_dict):
    def hash_smiles(smiles):
        return hashlib.md5(smiles.encode()).hexdigest()

    def create_conformer(mol, coordinates):
        mol = Chem.AddHs(mol)
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, (x, y, z) in enumerate(coordinates):
            conf.SetAtomPosition(i, Point3D(x, y, z))
        mol.AddConformer(conf)
        return mol

    cache_lut = {}

    def process_row(row):
        smiles = row["SMILES"]
        hash_key = hash_smiles(smiles)

        if hash_key in xyz_dict:
            mol = Chem.MolFromSmiles(smiles)
            _, coordinates = xyz_dict[hash_key]

            # Create 2D coordinates first
            AllChem.Compute2DCoords(mol)

            # Create a new conformer with the XYZ coordinates
            mol = create_conformer(mol, coordinates)

            # Generate the new SMILES
            if hash_key in cache_lut:
                new_smiles = cache_lut[hash_key]
            else:
                new_smiles = standardize_smiles(
                    Chem.MolToSmiles(mol, isomericSmiles=True)
                )
                cache_lut[hash_key] = new_smiles
            return new_smiles
        else:
            print(f"No matching XYZ file found for SMILES: {smiles}")
            print(f"Hash: {hash_key}")
            return None

    df["XTB_STANDARDIZED_SMILES"] = df.progress_apply(process_row, axis=1)

    print(f"Processed {len(df)} rows")
    print(
        f"Number of None values in XTB_STANDARDIZED_SMILES: {df['XTB_STANDARDIZED_SMILES'].isnull().sum()}"
    )

    return df


def run_xtb_optimization(input_dir, output_dir, xtb_path):
    """
    Run XTB optimization on all XYZ files in the input directory
    and save optimized structures to the output directory.

    Args:
        input_dir (str): Directory containing input XYZ files
        output_dir (str): Directory where optimized files will be saved
        xtb_path (str): Path to the xtb executable
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all XYZ files in input directory
    xyz_files = glob.glob(os.path.join(input_dir, "*.xyz"))

    # Process each XYZ file
    for xyz_path in xyz_files:
        try:
            # Get the filename without extension
            filename = Path(xyz_path).stem

            print(f"\nProcessing: {filename}")

            # Create temporary working directory for this optimization
            temp_dir = os.path.join(output_dir, f"temp_{filename}")
            os.makedirs(temp_dir, exist_ok=True)

            # Copy input XYZ to temporary directory
            temp_xyz = os.path.join(temp_dir, f"{filename}.xyz")
            shutil.copy2(xyz_path, temp_xyz)

            # Change to temporary directory
            original_dir = os.getcwd()
            os.chdir(temp_dir)

            # Run XTB optimization
            print("Running XTB optimization...")
            cmd = [xtb_path, f"{filename}.xyz", "--opt", "extreme"]

            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            if result.returncode != 0:
                print(f"Error during XTB optimization:\n{result.stderr}")
                continue

            # Check if optimization was successful
            if not os.path.exists("xtbopt.xyz"):
                print("XTB optimization failed - no xtbopt.xyz file produced")
                continue

            # Copy and rename the optimized structure
            optimized_name = f"{filename}_xtboptimized.xyz"
            output_path = os.path.join(output_dir, optimized_name)
            shutil.copy2("xtbopt.xyz", output_path)

            print(f"Successfully optimized: {optimized_name}")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

        finally:
            # Change back to original directory
            os.chdir(original_dir)

            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(
                    f"Warning: Could not remove temporary directory {temp_dir}: {str(e)}"
                )


def main(input_df_path, output_df_path, xtb_path):
    df = pd.read_csv(input_df_path)
    # Create output directory if it doesn't exist
    phase_1_dir = os.getcwd() + "/conformers/"
    phase_2_dir = (
        os.getcwd() + "/xtb_optimized"
    )  # Directory for optimized structures

    os.makedirs(phase_1_dir, exist_ok=True)
    os.makedirs(phase_2_dir, exist_ok=True)

    # Process each unique SMILES
    for smile in df["SMILES"].drop_duplicates():
        try:
            # Create molecule from SMILES
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                print(f"Failed to create molecule from SMILES: {smile}")
                continue

            Chem.SanitizeMol(mol)

            # Add hydrogens
            mol = Chem.AddHs(mol)

            # Generate 3D conformer
            success = AllChem.EmbedMolecule(mol, randomSeed=42)
            if success == -1:
                print(f"Failed to generate conformer for SMILES: {smile}")
                continue

            # Optimize with MMFF
            AllChem.MMFFOptimizeMolecule(mol)

            # Generate filename and save
            filename = generate_xyz_filename(smile)
            filepath = os.path.join(phase_1_dir, filename)
            save_mol_as_xyz(mol, filepath)

            print(f"Successfully processed: {smile} -> {filename}")

        except Exception as e:
            print(f"Error processing SMILES {smile}: {str(e)}")

    # Ensure XTB executable exists
    if not os.path.exists(xtb_path):
        print(f"Error: XTB executable not found at {xtb_path}")
        return

    # Run optimizations
    print("Starting XTB optimizations...")

    run_xtb_optimization(phase_1_dir, phase_2_dir, xtb_path)

    print("\nOptimization process completed!")

    print("\nReloading molecules from XTB output!")

    # NOTE: In our paper we had to reconstruct bond ordering from the xyz file. Since then we have found a better way: Instead we simply apply the updated atom positions to the pre-existing molecule.
    # NOTE: Some molecules may fail XTB optimization.
    xyz_dict = load_xyz_files(phase_2_dir)

    processed_df = process_molecules(df, xyz_dict)

    processed_df.to_csv(output_df_path)


if __name__ == "__main__":
    fire.Fire(main)
