import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, Lipinski, rdMolDescriptors
import py3Dmol
import streamlit.components.v1 as components

# Page Config
st.set_page_config(
    page_title="Cheminformatics Virtual Lab",
    page_icon="🧪",
    layout="wide"
)

# Header & Student Details
st.title("🧪 Cheminformatics 2D & 3D Chemical Structure Visualizer")

# Author Info Card
st.sidebar.markdown("---")
st.sidebar.subheader("👨‍💻 Student Details")
st.sidebar.write("**Name:** J. Joehaarish Shakil")
st.sidebar.write("**Reg No:** RA2611030050029")
st.sidebar.write("**Department:** Cyber Security")
st.sidebar.write("**Year:** First Year")
st.sidebar.write("**Institution:** SRM Institute of Science and Technology")
st.sidebar.markdown("---")

st.write("Enter a SMILES string below to view 2D/3D structures and molecular properties.")

# Sidebar input
smiles_input = st.sidebar.text_input("Enter SMILES String:", "CCO")

if smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    
    if mol:
        st.success("Valid SMILES Structure!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("2D Structure")
            img = Draw.MolToImage(mol)
            st.image(img, caption=f"SMILES: {smiles_input}", use_column_width=True)
            
        with col2:
            st.subheader("Molecular Properties")
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            
            st.write(f"**Molecular Weight:** {mw:.2f} g/mol")
            st.write(f"**LogP:** {logp:.2f}")
            st.write(f"**H-Bond Donors:** {hbd}")
            st.write(f"**H-Bond Acceptors:** {hba}")
    else:
        st.error("Invalid SMILES String. Please enter a valid chemical structure.")
