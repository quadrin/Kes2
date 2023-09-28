import streamlit as st
import streamlit_folium as folium
from tkinter import Tk
from tkinter import filedialog
from rdkit import Chem, DataStructs
from multiprocessing import Pool
from rdkit.DataStructs import CreateFromBitString
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem
from Bio.PDB import PDBList
from Bio.Data import IUPACData
from tmscoring import TMscoring
import uuid
#^ will need pip install iminuit, pip install Bio
from Bio import PDB
from Bio.SeqUtils import seq3, seq1
import shutil
import urllib.request
import io
from tmtools import tm_align
from tmtools.io import get_structure
from tmtools.io import get_residue_data as original_get_residue_data
from io import StringIO
import zipfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import base64
import re


#~~~~~~~~~~~~~~~~~~~Molecular Similarity Calculator~~~~~~~~~~~~~~~~~~~
def calculate_similarity_for_pair(args):
    i, j, fingerprints, similarity_function, metric = args
    if i == j:
        return 1.0
    else:
        fp1 = CreateFromBitString(fingerprints[i])
        fp2 = CreateFromBitString(fingerprints[j])
        if metric == "Tanimoto":
            return TanimotoSimilarity(fp1, fp2)
        else:
            x = fp1.GetNumOnBits()
            y = fp2.GetNumOnBits()
            z = len(set(fp1.GetOnBits()) & set(fp2.GetOnBits()))
            return similarity_function(z, x, y)

def calculate_braun_blanquet_similarity(x, y, z):
    return x / max(y, z)

def calculate_soergel_similarity(x, y, z):
    return (x + y + z) / (2 * x + y + z)

def calculate_manhattan_similarity(x, y, z):
    return (x + y + z) / (x + y + z + 1)

def calculate_asymmetric_similarity(x, y, z):
    return (x + y + z) / (x + y + z + 2)

def calculate_dot_product_similarity(x, y, z):
    return x

def calculate_simpson_similarity(x, y, z):
    return min(y, z) / x

def calculate_sokal_sneath_similarity(x, y, z):
    return (x + 2 * y + 2 * z) / (3 * x + 2 * y + 2 * z)

def calculate_tullos_similarity(x, y, z):
    return (x + y + z) / 3

def calculate_tversky_similarity(x, y, z):
    alpha = 1  # You can adjust the value of alpha
    beta = 1   # You can adjust the value of beta
    return x / (x + alpha * y + beta * z)

def calculate_dice_similarity(x, y, z):
    return (2 * x) / (x + y + z)

def calculate_cosine_similarity(x, y, z):
    return x / ((x + y) * (x + z)) ** 0.5

def calculate_kulczynski_similarity(x, y, z):
    return (min(x, y) + min(y, z)) / (x + y)

def calculate_mcconnaughey_similarity(x, y, z):
    return (x + y + z) / (x + y + z + 1)

def calculate_euclidean_distance(x, y, z):
    return ((x - y) ** 2 + (y - z) ** 2 + (z - x) ** 2) ** 0.5

def calculate_jaccard_similarity(x, y, z):
    return x / (x + y + z - x)

def calculate_hamming_distance(x, y, z):
    return (x + y + z) / 3

def calculate_canberra_distance(x, y, z):
    return (abs(x - y) + abs(y - z) + abs(z - x)) / (x + y + z)

def calculate_bray_curtis_dissimilarity(x, y, z):
    return (abs(x - y) + abs(y - z) + abs(z - x)) / (x + y + z)

def page_molecular_similarity():

    metric_functions = {
        "Braun Blanquet": calculate_braun_blanquet_similarity,
        "Soergel": calculate_soergel_similarity,
        "Manhattan": calculate_manhattan_similarity,
        "Asymmetric": calculate_asymmetric_similarity,
        "Dot-product": calculate_dot_product_similarity,
        "Simpson": calculate_simpson_similarity,
        "SokalSneath": calculate_sokal_sneath_similarity,
        "Tullos": calculate_tullos_similarity,
        "Tversky": calculate_tversky_similarity,
        "Dice": calculate_dice_similarity,
        "Cosine": calculate_cosine_similarity,
        "Kulczynski": calculate_kulczynski_similarity,
        "McConnaughey": calculate_mcconnaughey_similarity,
        "Euclidean": calculate_euclidean_distance,
        "Jacard": calculate_jaccard_similarity,
        "Hamming": calculate_hamming_distance,
        "Canberra": calculate_canberra_distance,
        "Bray Curtis": calculate_bray_curtis_dissimilarity,
        # Add other metrics here...
    }

    def zipdir(path, ziph):
        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file), 
                        os.path.relpath(os.path.join(root, file), 
                                        os.path.join(path, '..')))
                print(f"Added {file} to zip file")  # Print a message for each file

    def generate_heatmaps(selected_metrics, zipf):
        for metric in selected_metrics:
            # Check if the CSV file for this metric exists in the zip file
            try:
                csv_buffer = io.StringIO(zipf.read(f'similarity_scores/{metric}_similarity.csv').decode())
            except KeyError:
                print(f"CSV file for {metric} does not exist, skipping heatmap...")
                continue
            # Load the similarity scores from the CSV file
            similarity_df = pd.read_csv(csv_buffer, index_col=0)
            # Create a heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(similarity_df, cmap='viridis')
            # Write the heatmap to a PNG file in the zip file
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            plt.close()
            zipf.writestr(f'heatmaps/{metric}_heatmap.png', img_buffer.getvalue())

    def load_data(file):
        # Load the data from your CSV file into a pandas DataFrame
        try:
            data = pd.read_csv(file)
        except FileNotFoundError:
            print(f"File {file} not found.")
            return None
        except pd.errors.ParserError:
            print(f"File {file} is not a valid CSV.")
            return None

        if 'SMILES' not in data.columns or 'Compound' not in data.columns:
            print("Required columns ('SMILES', 'Compound') not found in the DataFrame.")
            return None

        return data

    def calculate_similarity(file, save_location, selected_metrics):
        data = load_data(file)
        if data is None:
            return

        smiles_strings = data['SMILES'].dropna().tolist()
        compound_ids = data['Compound'].loc[data['SMILES'].dropna().index].tolist()

        # Convert SMILES strings to RDKit Mol objects and calculate fingerprints
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_strings]
        fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024).ToBitString() for mol in mols if mol is not None]

        num_compounds = len(fingerprints)

        pool = Pool()  # creates a pool of worker processes

        for metric in selected_metrics:

            similarity_function = metric_functions.get(metric)
            if similarity_function is None:
                print(f"Unknown metric: {metric}")
                continue

            args = [(i, j, fingerprints, similarity_function, metric) for i in range(num_compounds) for j in range(num_compounds)]
            similarity_matrix = pool.map(calculate_similarity_for_pair, args)
            similarity_matrix = np.array(similarity_matrix).reshape(num_compounds, num_compounds)

            # Convert the matrix to a pandas DataFrame
            similarity_df = pd.DataFrame(similarity_matrix, index=compound_ids, columns=compound_ids)
            # Write the DataFrame to a CSV file in the zip file
            csv_buffer = io.StringIO()
            similarity_df.to_csv(csv_buffer)
            zipf.writestr(f'similarity_scores/{metric}_similarity.csv', csv_buffer.getvalue())

            print(f"Done with {metric}!")

    def download_link(object_to_download, download_filename, download_link_text):
        if isinstance(object_to_download,pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        b64 = base64.b64encode(object_to_download.encode()).decode()
        return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

    metrics = ["Tanimoto", "Dice", "Cosine", "Soergel", "Manhattan", "Euclidean", "Kulczynski", "McConnaughey", "Asymmetric", "Dot-product", "Simpson", "SokalSneath", "Tullos", "Tversky", "Braun Blanquet", "Jacard", "Hamming", "Canberra", "Bray Curtis"]

    st.title('Molecular Similarity Score Calculator')

    if st.button('Reset Session', key="resetSession"):
        st.session_state.clear()

    file = st.file_uploader('Upload File', type=['csv'], key='uploadMolecular')
    selected_metrics = st.multiselect('Select Metrics', metrics)

    if st.button('Select All Metrics'):
        selected_metrics = metrics

    if st.button('Deselect All Metrics'):  # Add this line
        selected_metrics = []

    save_location = st.text_input('Save Location', '', key="MolecularSave")  # Add this line

    if st.button('Run', key='RunButtonMolecular'):
        if file is not None and save_location:  # Check if save_location is not empty
            # Create a BytesIO object
            zip_buffer = io.BytesIO()

            # Create a Zip file
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                calculate_similarity(file, zipf, selected_metrics)
                generate_heatmaps(selected_metrics, zipf)

            # Create download button
            st.download_button(
            label="Download results",
            data=zip_buffer.getvalue(),
            file_name="results.zip",
            mime="application/zip",
            key="DownloadButtonMolecular"  # Add this line
        )
        else:
            st.error('Please upload a file and enter a save location or select an existing folder.')

#~~~~~~~~~~~~~~~~~~~Protein Similarity Calculator~~~~~~~~~~~~~~~~~~~

def extract_pdb_id(file_path):
    with open(file_path) as file:
        for line in file:
            if line.startswith("HEADER"):
                return line.split()[-1]
    return None

def get_residue_data(chain):
    try:
        return original_get_residue_data(chain)
    except KeyError as e:
        print(f"Encountered error: {e}. Skipping this residue.")
        # Handle the error here, possibly by returning a default value
        # or by modifying the chain to remove the offending residue
        # and then calling the original get_residue_data function again
        return [], []

# Add a global counter at the top of your script
counter = 0
pdb_files = []

def page_protein_similarity():
    global counter
    st.title('Protein Similarity Calculator')

    # Upload protein structure files
    protein_csv = st.file_uploader('Upload Protein CSV File', type=['csv'], key='uploadProtein')

    # Scoring methods
    scoring_methods = ['TM-score', 'RMSD', 'Other method']  # Add other methods here...
    selected_method = st.selectbox('Select Scoring Method', scoring_methods, key="SelectProtein")

    # Add a text input for the save location
    save_location = st.text_input('Save Location', '', key="ProteinUpload")  # Add this line

    # Initialize zip_buffer and temp_dir outside of the if block
    zip_buffer = io.BytesIO()
    temp_dir = None

    if protein_csv is not None:
        # Read the CSV file into a pandas DataFrame
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(protein_csv)

        # Get the protein names and PDB IDs from the first row of the DataFrame
        proteins = df.columns[1:]  # Get the protein names
        proteins = [protein.strip() for protein in proteins]
        pdb_ids = df.iloc[0, 1:]  # Get the PDB IDs

        # Create a mapping from protein names to PDB IDs
        protein_to_pdb = dict(zip(proteins, [pdb_id.upper() for pdb_id in pdb_ids]))

        print(f"protein_to_pdb: {protein_to_pdb}")
        print(f"proteins: {proteins}")

        # Button for downloading the PDB files
        if st.button('Download PDB Files', key='DownloadButtonProtein'):
            pdbl = PDBList()

            # Create a unique directory
            temp_dir = os.path.join(save_location, str(uuid.uuid4()))
            os.makedirs(temp_dir, exist_ok=True)

            # Initialize pdb_files in session state if it doesn't exist
            if 'pdb_files' not in st.session_state:
                st.session_state.pdb_files = []

            for pdb_id in pdb_ids:
                try:
                    pdb_file = pdbl.retrieve_pdb_file(pdb_id, pdir=temp_dir, file_format='pdb', overwrite=True)
                    st.session_state.pdb_files.append(pdb_file)  # Add the name of the PDB file to the list
                except Exception as e:  # Catch all exceptions
                    st.write(f"Error downloading PDB file for {pdb_id}: {str(e)}")

    if 'run_clicked' not in st.session_state:
        st.session_state.run_clicked = False

    if st.button('Run', key='RunButtonProtein'):
        st.session_state.run_clicked = True

    if st.session_state.run_clicked and protein_csv is not None:

        # Create a BytesIO object
        zip_buffer = io.BytesIO()

        # Create a Zip file
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Parse protein structures and calculate similarity
            st.write(f'Calculating protein similarity using {selected_method}...')
            # Initialize an empty DataFrame to store the similarity scores
            similarity_df = pd.DataFrame(index=proteins, columns=proteins)

            # Iterate over the PDB files stored in session state
            parser = PDB.PDBParser(QUIET=True)
            for pdb_file1 in st.session_state.pdb_files:
                try:
                    structure1 = parser.get_structure('protein2', pdb_file1)
                    chain1 = next(structure1.get_chains())
                    coords1, seq1 = get_residue_data(chain1)

                    pdb_id1 = extract_pdb_id(pdb_file1).upper()
                except KeyError:
                    st.write(f"Skipping {pdb_file1} due to non-standard residue")

                for pdb_file2 in st.session_state.pdb_files:
                    try:
                        structure2 = parser.get_structure('protein2', pdb_file2)
                        chain2 = next(structure2.get_chains())
                        coords2, seq2 = get_residue_data(chain2)

                        pdb_id2 = extract_pdb_id(pdb_file2).upper()
                    except KeyError:
                        st.write(f"Skipping {pdb_file2} due to non-standard residue")

                    # Get the protein names from the protein_to_pdb dictionary
                    try:
                        # Extract protein names directly from the file paths
                        protein1 = [protein for protein, pdb_id in protein_to_pdb.items() if pdb_id == pdb_id1][0]
                        protein2 = [protein for protein, pdb_id in protein_to_pdb.items() if pdb_id == pdb_id2][0]

                        if protein1 is None or protein2 is None:
                            raise ValueError(f"Protein names not found for files: {pdb_file1}, {pdb_file2}")

                    except ValueError as e:
                        print(str(e))
                        continue
                    
                    if proteins.index(protein1) <= proteins.index(protein2):
                            try:
                                if selected_method == 'TM-score':
                                    res = tm_align(coords1, coords2, seq1, seq2)
                                    score = res.tm_norm_chain1
                                elif selected_method == 'RMSD':
                                    res = tm_align(coords1, coords2, seq1, seq2)
                                    score = np.linalg.norm(coords1 - np.dot(coords2, res.u.T) - res.t, axis=1).mean()
                                else:
                                    # Implement other scoring methods here...
                                    pass
                            except Exception as e:
                                print(f"Error calculating score for {protein1} and {protein2}: {e}")
                                score = np.nan  # Use NaN as the score if an error occurs


                            # Store the score in the DataFrame
                            similarity_df.loc[protein1, protein2] = score
                            similarity_df.loc[protein2, protein1] = score

                            print(f"Protein1: {protein1}, PDB ID1: {pdb_id1}")
                            print(f"Protein2: {protein2}, PDB ID2: {pdb_id2}")
                            print(f"Score: {score}")
                            print(similarity_df)

        print(f"Size of zip_buffer: {zip_buffer.tell()}")

        # Write the DataFrame to a CSV file in the zip file
        csv_buffer = io.StringIO()
        similarity_df.to_csv(csv_buffer)
        zipf.writestr(f'similarity_scores/{selected_method}_similarity.csv', csv_buffer.getvalue())

        # Generate a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_df.astype(float), cmap='viridis')

        # Write the heatmap to a PNG file in the zip file
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        plt.close()
        zipf.writestr(f'heatmaps/{selected_method}_heatmap.png', img_buffer.getvalue())

    # Delete the temporary directory and all its contents if it's not None
    if temp_dir is not None:
        shutil.rmtree(temp_dir)

    # Create download button with a unique key for each function call
    st.download_button(
        label="Download results",
        data=zip_buffer.getvalue(),
        file_name="results.zip",
        mime="application/zip",
        key=f"DownloadButtonProtein_{counter}"  # Append the counter to the key
    )

    counter += 1  # Increment the counter
#~~~~~~~~~~~~~~~~~~~Page Menu~~~~~~~~~~~~~~~~~~~

# Define the pages
PAGES = {
    "Molecular Similarity Calculator": page_molecular_similarity,
    "Protein Similarity Calculator": page_protein_similarity
}

def main():
    # page_molecular_similarity()
    # page_protein_similarity()
    pass

if __name__ == "__main__":
    main()

# Create a sidebar for navigation
st.sidebar.title('Navigation')
selection = st.sidebar.selectbox("Go to", list(PAGES.keys()))

# Display the selected page
page = PAGES[selection]
page()

# if __name__ == "__main__":
#     main()

