from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
import io
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import numpy as np

def load_data(file):
    """
    Loads data from a CSV file into a pandas DataFrame.

    Parameters:
    file (str): The path to the CSV file.

    Returns:
    DataFrame: A pandas DataFrame containing the data from the CSV file.
    """
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

def tanimoto_similarity(query_fp, mol_fp):
    return DataStructs.TanimotoSimilarity(query_fp, mol_fp)

def rdkit_similarity(query_fp, mol_fp):
    return DataStructs.FingerprintSimilarity(query_fp, mol_fp)

def tversky_similarity(query_fp, mol_fp):
    alpha, beta = 0.5, 0.5
    return DataStructs.TverskySimilarity(query_fp, mol_fp, alpha, beta)

def euclidean_similarity(query_fp, mol_fp):
    return 1 - DataStructs.DiceSimilarity(query_fp, mol_fp)

def dice_similarity(query_fp, mol_fp):
    return DataStructs.DiceSimilarity(query_fp, mol_fp)

def cosine_similarity(query_fp, mol_fp):
    return DataStructs.CosineSimilarity(query_fp, mol_fp)

def rogot_goldberg_similarity(query_fp, mol_fp):
    return DataStructs.FingerprintSimilarity(query_fp, mol_fp, metric=DataStructs.DiceSimilarity)

# Define a mapping from metric names to their respective functions
metric_functions = {
    "Tanimoto": tanimoto_similarity,
    "rdkit": rdkit_similarity,
    "tversky": tversky_similarity,
    "euclidean": euclidean_similarity,
    "dice": dice_similarity,
    "cosine": cosine_similarity,
    "rogot_goldberg": rogot_goldberg_similarity
}

def calculate_similarity(args):
    method, query_fp, mol_fp, compound_ids, i, j = args
    if method in metric_functions:
        similarity = metric_functions[method](query_fp, mol_fp)
    else:
        raise ValueError(f"Unknown method: {method}")

    result_entry = {
        'Query Compound' : compound_ids[i],
        'Data Compound' : compound_ids[j],
        'Similarity' : similarity
    }
    return result_entry

def calculate_similarity_for_metrics(data, zipf, selected_metrics):
    smiles_strings = data['SMILES'].dropna().tolist()
    compound_ids = data['Compound'].loc[data['SMILES'].dropna().index].tolist()
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_strings]
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mols if mol is not None]
    num_compounds = len(fingerprints)

    for method in selected_metrics:
        with Pool() as pool:
            results = pool.map(calculate_similarity, [(method, fingerprints[i], fingerprints[j], compound_ids, i, j) for i in range(num_compounds) for j in range(i+1, num_compounds)])

        df = pd.DataFrame(results)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer)
        zipf.writestr(f'similarity_scores/{method}_similarity.csv', csv_buffer.getvalue())

def generate_heatmaps(selected_metrics, zipf):
    for metric in selected_metrics:
        # Check if the CSV file for this metric exists in the zip file
        try:
            csv_buffer = io.StringIO(zipf.read(f'similarity_scores/{metric}_similarity.csv').decode())
        except KeyError:
            print(f"CSV file for {metric} does not exist, skipping heatmap...")
            continue
        # Load the similarity scores from the CSV file
        similarity_df = pd.read_csv(csv_buffer)
        # Pivot the DataFrame to create a square matrix
        similarity_df = similarity_df.pivot_table(index='Query Compound', columns='Data Compound', values='Similarity')
        # Fill NaN values with 0
        similarity_df.fillna(0, inplace=True)
        # Create a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_df, cmap='viridis')
        # Write the heatmap to a PNG file in the zip file
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        plt.close()
        zipf.writestr(f'heatmaps/{metric}_heatmap.png', img_buffer.getvalue())
        
def download_link(buffer, filename, text):
    """
    Generates a link to download the zip file

    Parameters:
    buffer (io.BytesIO): The buffer that contains the zip file.
    filename (str): The name of the file to be downloaded.
    text (str): The text that will be displayed on the download link.

    Returns:
    str: a string that represents the HTML download link
    """
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:file/zip;base64,{b64}" download="{filename}">{text}</a>'
    return href