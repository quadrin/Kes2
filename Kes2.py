import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import io
import base64
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from utils import metric_functions, load_data, calculate_similarity_for_metrics, generate_heatmaps, download_link
import os

st.set_page_config(
        page_title="kes2",
)

# Constants
SIMILARITY_METRICS = list(metric_functions.keys())

def page_molecular_similarity():
    st.title('Kes2: Molecular Similarity Score Calculator')
    st.markdown('This app calculates the similarity scores between molecules in a CSV file.')

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    # Use the selected metrics from the session state in the multiselect widgets
    if 'selected_similarity_metrics' not in st.session_state:
        st.session_state['selected_similarity_metrics'] = []

    selected_similarity_metrics = st.multiselect("Select Similarity Metrics", list(metric_functions.keys()), st.session_state['selected_similarity_metrics'])

    # Add Select All / Deselect All buttons
    if st.button('Select All Similarity Metrics'):
        st.session_state['selected_similarity_metrics'] = list(metric_functions.keys())
        st.experimental_rerun()

    # Combine selected metrics
    selected_metrics = selected_similarity_metrics

    if st.button('Calculate Similarity'):
        if uploaded_file is not None:
            data = load_data(uploaded_file)
            if data is not None:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    calculate_similarity_for_metrics(data, zipf, selected_metrics)
                    generate_heatmaps(selected_metrics, zipf)

                st.download_button(
                    label="Download results",
                    data=zip_buffer.getvalue(),
                    file_name="results.zip",
                    mime="application/zip"
                )
        else:
            st.error('Please upload a file.')

logo_url = 'https://i.imgur.com/iZ4HlQF.png'

with st.sidebar:
  st.image(logo_url, width=200)

# Add ownership credit to the app
st.sidebar.markdown("""
---
## Kes2: Molecular Similarity Score Calculator 
https://github.com/quadrin/Kes2
                                
**Created by Alex Kesin. 
**Last updated on 01/13/24. For any questions, reach out to kesin@umich.edu.**
""")

# Add a new page for the demo/tutorial
# Add a new button to the sidebar for the demo/tutorial page
demo_button = st.sidebar.button('Demo/Tutorial', key = 'demo_button')

# Initialize the session state if it doesn't exist
if 'demo_button_clicked' not in st.session_state:
    st.session_state['demo_button_clicked'] = False

# Change the state when the button is clicked
if demo_button and not st.session_state['demo_button_clicked']:
    st.session_state['demo_button_clicked'] = True

# Change the state when the "Turn Demo Off" button is clicked
if st.sidebar.button('Turn Demo Off', key = 'turn_demo_off'):
    st.session_state['demo_button_clicked'] = False

# Display the demo/tutorial page if the button was clicked
if st.session_state['demo_button_clicked']:
    # Add explanations for each similarity metric
    st.markdown("""
    Further reading (credit @ Santu Chall): https://medium.com/@santuchal/understanding-molecular-similarity-51e8ebb38886

    ## Similarity Metrics
    ***Tanimoto Similarity:*** This metric measures the overlap of structural features between two chemical compounds using their binary fingerprints. A higher Tanimoto coefficient indicates a greater similarity, with values ranging from 0 to 1.

    ***RDKit Similarity:*** It generates a binary fingerprint for each molecule based on predefined molecular fragments or patterns.

    ***Tversky Similarity:*** An extension of the Jaccard index and Tanimoto coefficient, Tversky similarity allows for asymmetric comparisons of molecular structures. It uses two parameters (α and β) to emphasize different aspects of the molecules' features, offering a flexible approach to measuring structural similarity.

    ***Euclidean Similarity:*** This metric calculates the similarity between two molecules based on the inverse of the Euclidean distance between their molecular fingerprints or descriptors. A smaller distance indicates a higher similarity.

    ***Dice Similarity:*** Dice similarity assesses the overlap between the binary fingerprints of two molecules. It calculates the ratio of the shared features to the total number of features in both fingerprints, ranging from 0 (no overlap) to 1 (complete overlap).

    ***Cosine Similarity:*** This measure calculates the cosine of the angle between two fingerprint vectors, representing the directional agreement of their features. It ranges from -1 (completely dissimilar) to 1 (identical), with 0 indicating no structural correlation.

    ***Rogot-Goldberg Similarity:*** This metric focuses on atom environments within a certain radius from each atom in the molecular structure. It evaluates molecules based on the types and arrangements of atoms in their immediate vicinity.
    """)

    st.markdown('## Example CSV File Template')
    # Create a CSV template file
    csv_template = pd.DataFrame({
        'SMILES': ['OC1=CC(O)=C(C(N(C2=CC=C(C3=NC(Cl)=NC=C3C)C=C2)CC4=CC=C(CNC(C(F)F)=O)C=C4)=O)C=C1', 
                'OC1=C(C(N(CC2=CC=C(NC(C)=O)C=C2)C3=CC=C(C(O)=O)C=C3)=O)C=CC(O)=C1', 
                'OC1=C(C(N(CC2=CC=C(NC(C)=O)C=C2)C3=CC=C(C(NC4CC4)=O)C=C3)=O)C=CC(O)=C1', 
                'OC1=C(C(N(CC2=CC=C(NC(C)=O)C=C2)C3=CC=C(C(NC)=O)C=C3)=O)C=CC(O)=C1', 
                'OC1=C(C(N(CC2=CC=C(NC(C)=O)C=C2)C3=CC=C(C(NCCO)=O)C=C3)=O)C=CC(O)=C1', 
                'OC1=C(C(N(CC2=CC=C(NC(C)=O)C=C2)C3=CC=C(C(N(C)C)=O)C=C3)=O)C=CC(O)=C1'],
        'Compound': ['6', '14', '15a', '15b', '15c', '15d']
    })

    # Convert DataFrame to CSV
    csv_template_str = csv_template.to_csv(index=False)

    # Add a download button for the CSV template
    st.download_button(
        label="Download CSV Template",
        data=csv_template_str,
        file_name="template.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    page_molecular_similarity()
