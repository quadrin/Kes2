import streamlit as st
import io
import os
import uuid
import zipfile
import itertools
import concurrent.futures
import multiprocessing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.DataStructs import CreateFromBitString, TanimotoSimilarity
from Bio.PDB import PDBList, MMCIFParser, PDBIO, Select, PDBParser
from Bio import PDB
from tmtools import tm_align
from multiprocessing import Pool
from itertools import combinations
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
import numpy as np
from rdkit.Chem import AllChem

# Ignore BioPython PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)
#~~~~~~~~~~~~~~~~~~~Molecular Similarity Calculator~~~~~~~~~~~~~~~~~~~
def calculate_similarity_for_pair(args):
    i, j, fingerprints, similarity_function, metric = args
    if i == j:
        return 1.0
    else:
        fp1 = CreateFromBitString(fingerprints[i])
        fp2 = CreateFromBitString(fingerprints[j])
        if metric == "Tanimoto":
            if fp1 is not None and fp2 is not None:
                return TanimotoSimilarity(fp1, fp2)
        else:
            x = fp1.GetNumOnBits()
            y = fp2.GetNumOnBits()
            z = len(set(fp1.GetOnBits()) & set(fp2.GetOnBits()))
            return similarity_function(z, x, y)

def normalize(values):
    min_value = min(values)
    max_value = max(values)
    normalized_values = [(max_value - value) / (max_value - min_value) if value != 1.0 else 1.0 for value in values]
    return np.array(normalized_values)  # Convert the list to a numpy array

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

    def calculate_similarity(file, zipf, selected_metrics):
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

             # Normalize the similarity matrix
            similarity_matrix = normalize(similarity_matrix.flatten()).reshape(num_compounds, num_compounds)

            # Convert the matrix to a pandas DataFrame
            similarity_df = pd.DataFrame(similarity_matrix, index=compound_ids, columns=compound_ids)
            print(similarity_df)
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

    if 'selected_metrics' not in st.session_state:
        st.session_state.selected_metrics = []

    if st.button('Select All Metrics'):
        st.session_state.selected_metrics = metrics

    selected_metrics = st.session_state.selected_metrics

    if st.button('Deselect All Metrics'):  # Add this line
        selected_metrics = []

    if st.button('Run', key='RunButtonMolecular'):
        if file is not None:  # Check if save_location is not empty
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
                mime="application/zip"
            )
        else:
            st.error('Please upload a file.')

#~~~~~~~~~~~~~~~~~~~Protein Similarity Calculator~~~~~~~~~~~~~~~~~~~

progress_bar = st.progress(0)

def extract_pdb_id(file):
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

selected_method = None

class AlphaCarbonSelect(Select):
    """Only accept alpha carbon atoms."""
    def accept_atom(self, atom):
        return atom.get_name() == "CA"

parser = PDB.MMCIFParser()
pdb_io = PDB.PDBIO()

def get_tm_score(pdb_file1, pdb_file2):
    # Parse the protein structures
    structure1 = PDBParser().get_structure('protein1', pdb_file1)
    structure2 = PDBParser().get_structure('protein2', pdb_file2)

    # Extract the alpha carbon atoms from the Structure objects
    builder = CaPPBuilder()
    for chain1 in structure1.get_chains():
        for chain2 in structure2.get_chains():
            coords1, seq1 = get_residue_data(chain1)
            coords2, seq2 = get_residue_data(chain2)

            # Calculate the TM-score
            res = tm_align(coords1, coords2, seq1, seq2)
            tm_score = res.tm_norm_chain1  # or res.tm_norm_chain2 depending on which normalization you want

    return tm_score

def process_pair_wrapper(args):
    return process_pair(*args)

def process_pair(pdb_file1, pdb_file2, use_simplified, selected_method):
    similarity = float('inf')  # Assign a default value to similarity
    print(f"Processing pair: {pdb_file1}, {pdb_file2}")  # Print the pair being processed
    try:
        # Parse the protein structures
        print("Parsing protein structures...")
        structure1 = parser.get_structure('protein1', io.StringIO(pdb_file1))
        structure2 = parser.get_structure('protein2', io.StringIO(pdb_file2))

        # Preprocess the structures if use_simplified is True
        if use_simplified:
            print("Preprocessing structures...")
            try:
                # Generate unique filenames
                file1_name = f'structure1_{uuid.uuid4()}.cif'
                file2_name = f'structure2_{uuid.uuid4()}.cif'

                pdb_io = PDBIO()
                pdb_io.set_structure(structure1)
                pdb_io.save(file1_name, AlphaCarbonSelect())
                pdb_io.set_structure(structure2)
                pdb_io.save(file2_name, AlphaCarbonSelect())

                # Reload the preprocessed structures
                structure1 = parser.get_structure('protein1', file1_name)
                structure2 = parser.get_structure('protein2', file2_name)
            except Exception as e:
                print(f"Error during preprocessing: {e}")
            finally:
                # Clean up the files
                if os.path.exists(file1_name):
                    os.remove(file1_name)
                if os.path.exists(file2_name):
                    os.remove(file2_name)

        # Calculate similarity
        print("Calculating similarity...")
        if selected_method == 'TM-score':
                 # Use TM-align for TM-score calculation
                print("Calculating TM-score...")
                # Parse the protein structures
                structure1 = PDBParser().get_structure('protein1', pdb_file1)
                structure2 = PDBParser().get_structure('protein2', pdb_file2)
                # Extract the alpha carbon atoms from the Structure objects
                builder = CaPPBuilder()
                tm_scores = []
                for chain1 in structure1.get_chains():
                    for chain2 in structure2.get_chains():
                        ca_list1 = [res["CA"] for res in chain1.get_residues() if res.has_id("CA")]
                        ca_list2 = [res["CA"] for res in chain2.get_residues() if res.has_id("CA")]
                        # Check if both lists are not empty
                        if ca_list1 and ca_list2:
                            # Align the chains
                            super_imposer = Superimposer()
                            super_imposer.set_atoms(ca_list1, ca_list2)
                            super_imposer.apply(chain2.get_atoms())
                            # Calculate the TM-score
                            tm_score = get_tm_score(ca_list1, ca_list2)
                            tm_scores.append(tm_score)
                # Calculate the average TM-score
                similarity = sum(tm_scores) / len(tm_scores)
                progress = (i + 1) / len(pairs)
                progress_bar.progress(progress)

        elif selected_method == 'RMSD':
            # Use ProDy for RMSD calculation
            print("Calculating RMSD...")
            transformation, rmsd = calcTransformation(structure1, structure2)
            similarity = rmsd
            progress = (i + 1) / len(pairs)
            progress_bar.progress(progress)

        print(f"Calculated similarity: {similarity}")  # Print the calculated similarity
    
    except ValueError as e:
        print(f"Error processing pair {pdb_file1}, {pdb_file2}: {str(e)}. Skipping this pair.")
        return (pdb_file1, pdb_file2, -1)  # Return a default similarity score
    
    return (pdb_file1, pdb_file2, similarity)

def page_protein_similarity():
    global selected_method
    
    st.title('Protein Similarity Calculator')

    if st.button('Reset Session', key="resetSessionPtn"):
        st.session_state.clear()

    if 'pdb_files' not in st.session_state:
        st.session_state.pdb_files = []

    protein_csv = st.file_uploader('Upload Protein CSV File', type=['csv'], key='uploadProtein')
    scoring_methods = ['TM-score', 'RMSD']  # Add other methods here...
    selected_method = st.selectbox('Select Scoring Method', scoring_methods, key="SelectProtein")

    if protein_csv is not None:
        df = pd.read_csv(protein_csv)
        proteins = df.columns[1:]  # Get the protein names
        proteins = [protein.strip() for protein in proteins]
        pdb_ids = df.iloc[0, 1:]  # Get the PDB IDs
        protein_to_pdb = dict(zip(proteins, [pdb_id.upper() for pdb_id in pdb_ids]))

        if st.button('Load PDB Files', key='DownloadButtonProtein'):
            pdbl = PDBList()
            for pdb_id in pdb_ids:
                try:
                    pdb_file_path = pdbl.retrieve_pdb_file(pdb_id)
                    with open(pdb_file_path, 'r') as f:
                        pdb_file = f.read()
                    st.session_state.pdb_files.append(pdb_file)  # Add the PDB file to the list
                except Exception as e:  # Catch all exceptions
                    st.write(f"Error downloading PDB file for {pdb_id}: {str(e)}")

    use_simplified = st.checkbox('Use simplified representation (alpha carbon atoms only)', value=False)

    if st.button('View PDB Files'):
        for i, pdb_file in enumerate(st.session_state.pdb_files):
            st.subheader(f'PDB File {i+1}')
            st.text(pdb_file)

    if st.button('Run', key='RunButtonProtein'):
        if protein_csv is not None: 
            # Create a BytesIO object
            zip_buffer = io.BytesIO()

            # Create a Zip file
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Parse protein structures and calculate similarity
                print(f'Calculating protein similarity using {selected_method}...')
                # Initialize an empty DataFrame to store the similarity scores
                similarity_df = pd.DataFrame(index=proteins, columns=proteins)

                # Create a pool of worker processes
                with Pool() as pool:
                    pairs = [(pdb_file1, pdb_file2, use_simplified, selected_method) for pdb_file1, pdb_file2 in combinations(st.session_state.pdb_files, 2)]
                    print("Starting multiprocessing...")
                    results = []
                    try:
                        results = pool.map(process_pair_wrapper, pairs)
                    except Exception as e:
                        print(f"Error during multiprocessing: {str(e)}")
                                    
                # Add your code to store the similarities in similarity_df
                print("Storing results...")
                for pdb_file1, pdb_file2, similarity in results:
                    protein1 = extract_pdb_id(pdb_file1)
                    protein2 = extract_pdb_id(pdb_file2)
                    similarity_df.loc[protein1, protein2] = similarity
                    similarity_df.loc[protein2, protein1] = similarity  # if the similarity is symmetric

                # Write the DataFrame to a CSV file in the zip file
                csv_buffer = io.StringIO()
                similarity_df.to_csv(csv_buffer)
                zipf.writestr(f'similarity_scores/{selected_method}_similarity.csv', csv_buffer.getvalue())

                # Generate a heatmap
                print("Generating heatmap...")
                plt.figure(figsize=(10, 8))
                sns.heatmap(similarity_df.astype(float), cmap='viridis')

                # Write the heatmap to a PNG file in the zip file
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png')
                plt.close()
                zipf.writestr(f'heatmaps/{selected_method}_heatmap.png', img_buffer.getvalue())

            # Create download button
            st.download_button(
                label="Download results",
                data=zip_buffer.getvalue(),
                file_name="results.zip",
                mime="application/zip"
            )
        else:
            st.error('Please upload a file.')
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

