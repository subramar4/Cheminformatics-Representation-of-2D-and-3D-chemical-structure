"""
Cheminformatics Virtual Laboratory — enhanced Streamlit app
Run: streamlit run app.py

Install dependencies:
pip install streamlit pandas matplotlib rdkit py3Dmol reportlab numpy
"""

from io import BytesIO
import tempfile
import os
import random

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from rdkit import Chem
from rdkit.Chem import (
    Draw, AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors,
    QED
)
import py3Dmol
import streamlit.components.v1 as components

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as PDFImage
)
from reportlab.lib.units import inch


# ---------------------------------------------------------------------
# PAGE SETUP + STYLE
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Cheminformatics Virtual Laboratory",
    page_icon="🧪",
    layout="wide",
)

st.markdown("""
<style>
.main-title { text-align:center; font-size:2.25rem; font-weight:800; }
.subtle { color:#667085; text-align:center; }
div[data-testid="stMetric"] { border:1px solid #e6eaf0; padding:12px; border-radius:12px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧪 Cheminformatics Virtual Laboratory</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Explore molecular structures, properties, drug-likeness and chemistry concepts</div>', unsafe_allow_html=True)
st.divider()


# ---------------------------------------------------------------------
# EXAMPLE MOLECULES
# SMILES are intended as educational examples; RDKit validates them.
# ---------------------------------------------------------------------
EXAMPLES = {
    "Ethanol": "CCO",
    "Methane": "C",
    "Acetic acid": "CC(=O)O",
    "Acetone": "CC(=O)C",
    "Benzene": "c1ccccc1",
    "Toluene": "Cc1ccccc1",
    "Phenol": "Oc1ccccc1",
    "Aniline": "Nc1ccccc1",
    "Glucose (open-chain)": "C(C1C(C(C(C(O1)O)O)O)O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Nicotine": "CN1CCC[C@H]1c2cccnc2",
    "Lactic acid": "CC(O)C(=O)O",
    "Glycine": "NCC(=O)O",
    "Alanine": "CC(N)C(=O)O",
    "Urea": "NC(=O)N",
    "Limonene": "CC1=CCC(CC1)C(=C)C",
    "Naphthalene": "c1ccc2ccccc2c1",
    "Ciprofloxacin": "O=C(O)C1=CN(C2CC2)C2=CC(N3CCNCC3)=C(F)C=C2C1=O",
}


# ---------------------------------------------------------------------
# CORE CHEMISTRY HELPERS
# ---------------------------------------------------------------------
def parse_smiles(smiles: str):
    """Return an RDKit molecule, or None if SMILES is invalid."""
    if not smiles or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def bond_information(mol):
    """
    Count bonds using a conventional Lewis-structure interpretation:
    - heavy-atom bonds: bonds in the input molecule graph
    - sigma bonds: all atom-to-atom connections after adding H atoms
    - pi bonds: extra bond-order components after Kekulization
      (double contributes 1 pi, triple contributes 2 pi)
    Aromatic molecules are Kekulized where possible.
    """
    heavy_atom_bonds = mol.GetNumBonds()
    mol_h = Chem.AddHs(Chem.Mol(mol))
    sigma_bonds = mol_h.GetNumBonds()

    pi_bonds = 0
    kekule = Chem.Mol(mol)
    try:
        Chem.Kekulize(kekule, clearAromaticFlags=True)
        for bond in kekule.GetBonds():
            bt = bond.GetBondType()
            if bt == Chem.BondType.DOUBLE:
                pi_bonds += 1
            elif bt == Chem.BondType.TRIPLE:
                pi_bonds += 2
            elif bt == Chem.BondType.QUADRUPLE:
                pi_bonds += 3
    except Exception:
        # Aromatic fallback: each aromatic ring has a delocalized pi system.
        # Half a pi bond per aromatic graph edge is a useful graph estimate.
        pi_bonds = sum(0.5 for b in mol.GetBonds() if b.GetIsAromatic())

    return {
        "Heavy-atom bonds": heavy_atom_bonds,
        "Sigma bonds (incl. H)": sigma_bonds,
        "Pi bonds (estimated)": round(pi_bonds, 2),
        "Total sigma + pi": round(sigma_bonds + pi_bonds, 2),
        "Atoms including H": mol_h.GetNumAtoms(),
        "Heavy atoms": mol.GetNumHeavyAtoms(),
        "Total atoms (input graph)": mol.GetNumAtoms(),
    }


def calculate_properties(mol):
    """Calculate common RDKit descriptors and educational bond counts."""
    binfo = bond_information(mol)
    props = {
        "Molecular formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular weight (g/mol)": round(Descriptors.MolWt(mol), 3),
        "Exact molecular weight (Da)": round(Descriptors.ExactMolWt(mol), 5),
        "Canonical SMILES": Chem.MolToSmiles(mol, canonical=True),
        "LogP (Crippen)": round(Crippen.MolLogP(mol), 3),
        "TPSA (Å²)": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "H-bond donors": int(Lipinski.NumHDonors(mol)),
        "H-bond acceptors": int(Lipinski.NumHAcceptors(mol)),
        "Rotatable bonds": int(Lipinski.NumRotatableBonds(mol)),
        "Ring count": int(Lipinski.RingCount(mol)),
        "Aromatic rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "Aliphatic rings": int(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        "Fraction Csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "Molar refractivity": round(Crippen.MolMR(mol), 3),
        "Heavy atom count": int(mol.GetNumHeavyAtoms()),
        "Formal charge": int(Chem.GetFormalCharge(mol)),
        "QED drug-likeness (0–1)": round(QED.qed(mol), 3),
    }
    props.update(binfo)
    return props


def lipinski_analysis(props):
    """Apply the common Rule of Five thresholds as a screening heuristic."""
    checks = {
        "Molecular weight ≤ 500": props["Molecular weight (g/mol)"] <= 500,
        "LogP ≤ 5": props["LogP (Crippen)"] <= 5,
        "H-bond donors ≤ 5": props["H-bond donors"] <= 5,
        "H-bond acceptors ≤ 10": props["H-bond acceptors"] <= 10,
    }
    violations = sum(not passed for passed in checks.values())
    return checks, violations


def make_2d_image(mol, size=(850, 600), highlight_atoms=None):
    """Render a clear 2D structure with atom indices optionally highlighted."""
    draw_mol = Chem.Mol(mol)
    opts = Draw.MolDrawOptions()
    opts.addAtomIndices = True
    opts.bondLineWidth = 2
    opts.padding = 0.12
    return Draw.MolToImage(
        draw_mol, size=size, options=opts,
        highlightAtoms=highlight_atoms or []
    )


def make_3d_molecule(mol):
    """Generate and optimize a 3D conformer with MMFF, then UFF fallback."""
    mol3d = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol3d, params)
    if status != 0:
        return None
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol3d):
            AllChem.MMFFOptimizeMolecule(mol3d, maxIters=300)
        else:
            AllChem.UFFOptimizeMolecule(mol3d, maxIters=300)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol3d, maxIters=200)
        except Exception:
            pass
    return mol3d


def render_3d(mol3d, style_name="Ball and stick", show_surface=False):
    """Render interactive 3D model with py3Dmol."""
    mol_block = Chem.MolToMolBlock(mol3d)
    view = py3Dmol.view(width=900, height=560)
    view.addModel(mol_block, "mol")
    styles = {
        "Ball and stick": {"stick": {"radius": 0.14}, "sphere": {"scale": 0.28}},
        "Stick": {"stick": {"radius": 0.18}},
        "Space filling": {"sphere": {"scale": 0.85}},
        "Wireframe": {"line": {"linewidth": 2}},
    }
    view.setStyle(styles.get(style_name, styles["Ball and stick"]))
    if show_surface:
        view.addSurface(py3Dmol.VDW, {"opacity": 0.35, "color": "lightblue"})
    view.setBackgroundColor("#ffffff")
    view.zoomTo()
    components.html(view._make_html(), height=590, scrolling=False)


def create_property_chart(properties, keys):
    values = []
    labels = []
    for key in keys:
        value = properties.get(key)
        if isinstance(value, (int, float)):
            labels.append(key)
            values.append(float(value))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if values:
        ax.barh(labels, values)
        ax.set_xlabel("Value (different units; compare within a property only)")
        ax.set_title("Selected molecular descriptors")
        ax.invert_yaxis()
        fig.tight_layout()
    return fig


def generate_pdf_report(name, smiles, mol, props, checks, violations):
    """Build PDF report with a 2D image, property table and descriptor chart."""
    buffer = BytesIO()
    with tempfile.TemporaryDirectory() as tmpdir:
        structure_path = os.path.join(tmpdir, "structure.png")
        chart_path = os.path.join(tmpdir, "properties.png")

        make_2d_image(mol, size=(900, 620)).save(structure_path)
        chart_keys = [
            "Molecular weight (g/mol)", "LogP (Crippen)", "TPSA (Å²)",
            "Rotatable bonds", "H-bond donors", "H-bond acceptors"
        ]
        fig = create_property_chart(props, chart_keys)
        fig.savefig(chart_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        doc = SimpleDocTemplate(
            buffer, pagesize=A4, rightMargin=38, leftMargin=38,
            topMargin=38, bottomMargin=38
        )
        styles = getSampleStyleSheet()
        title = ParagraphStyle(
            "CVLTitle", parent=styles["Title"], fontSize=19,
            alignment=TA_CENTER, textColor=HexColor("#17365D"), spaceAfter=12
        )
        heading = ParagraphStyle(
            "CVLHeading", parent=styles["Heading2"], fontSize=13,
            textColor=HexColor("#17365D"), spaceBefore=10, spaceAfter=6
        )
        body = styles["BodyText"]
        story = [
            Paragraph("CHEMINFORMATICS VIRTUAL LABORATORY", title),
            Paragraph(f"Molecular Report: {name}", heading),
            Paragraph(f"<b>Input SMILES:</b> {smiles}", body),
            Paragraph(f"<b>Canonical SMILES:</b> {props['Canonical SMILES']}", body),
            Spacer(1, 8),
            Paragraph("2D Molecular Structure (atom indices shown)", heading),
            PDFImage(structure_path, width=5.8*inch, height=4.0*inch),
            Paragraph("Molecular Properties", heading),
        ]

        table_data = [["Property", "Value"]]
        for key, value in props.items():
            table_data.append([str(key), str(value)])
        table = Table(table_data, colWidths=[3.6*inch, 2.1*inch], repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HexColor("#F3F6FA")]),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.extend([table, Spacer(1, 10), Paragraph("Drug-likeness: Lipinski Rule of Five", heading)])
        lip_data = [["Criterion", "Pass?"]] + [[k, "Pass" if v else "Violation"] for k, v in checks.items()]
        lip_table = Table(lip_data, colWidths=[4.5*inch, 1.2*inch])
        lip_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F5597")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.extend([
            lip_table,
            Paragraph(f"Total Rule of Five violations: {violations}", body),
            Spacer(1, 8),
            Paragraph("Selected Descriptor Graph", heading),
            PDFImage(chart_path, width=5.9*inch, height=2.7*inch),
            Spacer(1, 8),
            Paragraph(
                "Interpretation note: descriptors and Lipinski screening are computational "
                "estimates. Passing these thresholds does not establish safety, efficacy, "
                "absorption, or suitability as a medicine. Results depend on the structure "
                "and the software implementation.",
                body
            ),
        ])
        doc.build(story)
    return buffer.getvalue()


# ---------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------
if "active_smiles" not in st.session_state:
    st.session_state.active_smiles = EXAMPLES["Ethanol"]
if "quiz_index" not in st.session_state:
    st.session_state.quiz_index = 0
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0
if "quiz_answered" not in st.session_state:
    st.session_state.quiz_answered = False


# ---------------------------------------------------------------------
# SIDEBAR: SHARED MOLECULE INPUT
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Molecule input")
    example_name = st.selectbox("Load an example", list(EXAMPLES.keys()))
    if st.button("Load selected molecule", use_container_width=True):
        st.session_state.active_smiles = EXAMPLES[example_name]
    st.text_area(
        "SMILES notation",
        key="active_smiles",
        height=90,
        help="Paste a SMILES string or load an example above."
    )
    st.caption("The same molecule input is used by the Explorer and PDF report.")

mol = parse_smiles(st.session_state.active_smiles)
if mol is None:
    st.error("This SMILES string could not be parsed. Try one of the example molecules.")
    st.stop()

props = calculate_properties(mol)
checks, violations = lipinski_analysis(props)
display_name = next((name for name, smi in EXAMPLES.items()
                     if parse_smiles(smi) is not None and
                     Chem.MolToSmiles(parse_smiles(smi)) == Chem.MolToSmiles(mol)),
                    "Custom molecule")

tabs = st.tabs([
    "🔬 Molecule Explorer", "⚖️ Compare Molecules",
    "💊 Drug-likeness", "📄 PDF Report", "🎓 Learning Mode"
])


# ---------------------------------------------------------------------
# TAB 1: EXPLORER
# ---------------------------------------------------------------------
with tabs[0]:
    st.subheader(f"Molecule Explorer — {display_name}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Formula", props["Molecular formula"])
    m2.metric("Molecular weight", f'{props["Molecular weight (g/mol)"]} g/mol')
    m3.metric("Heavy atoms", props["Heavy atom count"])
    m4.metric("QED score", props["QED drug-likeness (0–1)"])

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### 2D structure")
        st.image(make_2d_image(mol), use_container_width=True)
        st.caption("Atom indices are shown to help with atom-by-atom discussion.")
    with right:
        st.markdown("#### Interactive 3D structure")
        style = st.selectbox("Display style", ["Ball and stick", "Stick", "Space filling", "Wireframe"])
        surface = st.checkbox("Show van der Waals surface", value=False)
        mol3d = make_3d_molecule(mol)
        if mol3d is None:
            st.warning("A 3D conformer could not be generated for this structure.")
        else:
            render_3d(mol3d, style, surface)
            st.caption("The 3D model is a generated conformer, not an experimentally determined structure.")

    st.markdown("#### Molecular properties")
    property_df = pd.DataFrame(
        [{"Property": k, "Value": v} for k, v in props.items()]
    )
    st.dataframe(property_df, use_container_width=True, hide_index=True)

    st.markdown("#### Bond and atom counts")
    bond_keys = [
        "Total atoms (input graph)", "Atoms including H", "Heavy atoms",
        "Heavy-atom bonds", "Sigma bonds (incl. H)", "Pi bonds (estimated)",
        "Total sigma + pi"
    ]
    cols = st.columns(4)
    for i, key in enumerate(bond_keys):
        cols[i % 4].metric(key, props[key])
    st.info(
        "Counting convention: sigma bonds are atom-to-atom connections after explicit "
        "hydrogens are added. Pi bonds count additional bond-order components in a "
        "Kekulé structure. Aromatic fallback values are estimates because aromatic "
        "electrons are delocalized."
    )

    chart_keys = st.multiselect(
        "Choose numeric descriptors for a graph",
        [
            "Molecular weight (g/mol)", "Exact molecular weight (Da)",
            "LogP (Crippen)", "TPSA (Å²)", "H-bond donors",
            "H-bond acceptors", "Rotatable bonds", "Ring count",
            "Fraction Csp3", "Molar refractivity", "QED drug-likeness (0–1)"
        ],
        default=["Molecular weight (g/mol)", "LogP (Crippen)", "TPSA (Å²)", "Rotatable bonds"]
    )
    if chart_keys:
        fig = create_property_chart(props, chart_keys)
        st.pyplot(fig)
        plt.close(fig)


# ---------------------------------------------------------------------
# TAB 2: COMPARE
# ---------------------------------------------------------------------
with tabs[1]:
    st.subheader("Compare two molecules")
    c1, c2 = st.columns(2)
    with c1:
        name_a = st.selectbox("Molecule A", list(EXAMPLES.keys()), index=0, key="cmp_a")
    with c2:
        name_b = st.selectbox("Molecule B", list(EXAMPLES.keys()), index=9, key="cmp_b")
    ma, mb = parse_smiles(EXAMPLES[name_a]), parse_smiles(EXAMPLES[name_b])
    if ma and mb:
        ca, cb = st.columns(2)
        with ca:
            st.markdown(f"**{name_a}**")
            st.image(make_2d_image(ma, (600, 400)), use_container_width=True)
        with cb:
            st.markdown(f"**{name_b}**")
            st.image(make_2d_image(mb, (600, 400)), use_container_width=True)
        pa, pb = calculate_properties(ma), calculate_properties(mb)
        compare_keys = [
            "Molecular formula", "Molecular weight (g/mol)", "LogP (Crippen)",
            "TPSA (Å²)", "H-bond donors", "H-bond acceptors", "Rotatable bonds",
            "Ring count", "Heavy atom count", "QED drug-likeness (0–1)"
        ]
        comp = pd.DataFrame({
            "Property": compare_keys,
            name_a: [pa[k] for k in compare_keys],
            name_b: [pb[k] for k in compare_keys],
        })
        st.dataframe(comp, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# TAB 3: DRUG-LIKENESS
# ---------------------------------------------------------------------
with tabs[2]:
    st.subheader("Drug-likeness screening")
    st.write(f"**Current molecule:** {display_name} · **QED:** {props['QED drug-likeness (0–1)']}")
    st.write(f"**Lipinski Rule of Five violations:** {violations} out of 4")
    for criterion, passed in checks.items():
        st.write(("✅ " if passed else "⚠️ ") + criterion)
    if violations == 0:
        st.success("This molecule meets all four common Rule of Five thresholds.")
    elif violations == 1:
        st.warning("This molecule has one Rule of Five threshold violation.")
    else:
        st.warning(f"This molecule has {violations} Rule of Five threshold violations.")
    st.markdown("#### What these results mean")
    st.markdown("""
    - **Molecular weight:** a common threshold is 500 g/mol.
    - **LogP:** estimates partitioning between non-polar and aqueous environments.
    - **H-bond donors:** typically counts donor groups such as suitable N–H and O–H groups.
    - **H-bond acceptors:** counts atoms that can accept hydrogen bonds under the descriptor rules.
    - **QED:** a composite drug-likeness score from 0 to 1; it is not a probability of success.
    """)
    st.caption(
        "Lipinski's Rule of Five is a heuristic often used in early oral-drug screening. "
        "It does not establish bioavailability, toxicity, clinical effectiveness, or safety."
    )


# ---------------------------------------------------------------------
# TAB 4: PDF REPORT
# ---------------------------------------------------------------------
with tabs[3]:
    st.subheader("Create a downloadable PDF report")
    report_name = st.text_input("Report title / molecule name", value=display_name)
    st.write("The PDF includes a 2D structure, molecular descriptors, bond counts, Lipinski analysis and a graph.")
    if st.button("Generate PDF report", type="primary"):
        try:
            pdf = generate_pdf_report(
                report_name.strip() or "Molecule",
                st.session_state.active_smiles,
                mol, props, checks, violations
            )
            st.download_button(
                "⬇️ Download molecular report (PDF)",
                data=pdf,
                file_name="cheminformatics_molecular_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            st.success("PDF generated. Use the download button above.")
        except Exception as exc:
            st.error(f"Could not create the PDF: {exc}")


# ---------------------------------------------------------------------
# TAB 5: LEARNING MODE + QUIZ
# ---------------------------------------------------------------------
with tabs[4]:
    st.subheader("Learn cheminformatics")
    lesson = st.selectbox(
        "Choose a mini-lesson",
        ["SMILES", "Molecular weight", "LogP", "TPSA", "Hydrogen bonding",
         "Sigma and pi bonds", "Lipinski Rule of Five", "QED"]
    )
    lessons = {
        "SMILES": "SMILES is a text notation for encoding a molecular structure. For example, CCO represents ethanol. Validity depends on correct syntax and chemical valence.",
        "Molecular weight": "Molecular weight is the sum of the average atomic weights of atoms in a molecule. Exact molecular weight uses exact isotopic masses and is often used in mass spectrometry.",
        "LogP": "LogP estimates how a neutral compound partitions between octanol and water. Higher values generally indicate greater lipophilicity; ionization and pH can change real-world behavior.",
        "TPSA": "Topological polar surface area estimates the surface contribution from polar atoms, especially oxygen and nitrogen. It is one descriptor used when discussing permeability.",
        "Hydrogen bonding": "Hydrogen-bond donors provide a suitable hydrogen, often from O–H or N–H. Acceptors have suitable lone pairs. RDKit applies defined counting rules to these groups.",
        "Sigma and pi bonds": "A sigma bond is the primary bond along the internuclear axis. Double bonds generally contain one sigma and one pi bond; triple bonds contain one sigma and two pi bonds. Aromatic bonding is delocalized.",
        "Lipinski Rule of Five": "A common oral-drug screening heuristic uses molecular weight ≤500, LogP ≤5, no more than 5 H-bond donors and no more than 10 H-bond acceptors. It is not a guarantee of drug success.",
        "QED": "Quantitative Estimate of Drug-likeness combines several molecular properties into a score between 0 and 1. It is a heuristic, not a safety or efficacy score."
    }
    st.info(lessons[lesson])

    st.markdown("#### Quick chemistry quiz")
    quiz = [
        {
            "q": "Which SMILES string represents ethanol in this app's examples?",
            "options": ["CCO", "c1ccccc1", "CC(=O)O", "C"],
            "answer": "CCO",
            "explain": "CCO encodes a two-carbon chain with an oxygen at the end, corresponding to ethanol."
        },
        {
            "q": "A carbon–carbon double bond generally contains:",
            "options": ["Two sigma bonds", "One sigma and one pi bond", "Two pi bonds", "One pi bond only"],
            "answer": "One sigma and one pi bond",
            "explain": "A double bond consists of one sigma bond and one pi bond."
        },
        {
            "q": "Which descriptor estimates topological polar surface area?",
            "options": ["TPSA", "LogP", "QED", "Fraction Csp3"],
            "answer": "TPSA",
            "explain": "TPSA is the topological polar surface area descriptor."
        },
        {
            "q": "Which is a standard Lipinski Rule of Five threshold?",
            "options": ["Molecular weight ≤ 500 g/mol", "Molecular weight ≤ 50 g/mol", "TPSA must equal zero", "Exactly zero rotatable bonds"],
            "answer": "Molecular weight ≤ 500 g/mol",
            "explain": "One common threshold is molecular weight no greater than 500 g/mol."
        },
        {
            "q": "What does a QED score represent in this app?",
            "options": ["A composite drug-likeness estimate", "A direct toxicity measurement", "The number of atoms", "The molecule's 3D coordinates"],
            "answer": "A composite drug-likeness estimate",
            "explain": "QED combines multiple molecular descriptors into a drug-likeness estimate; it is not a direct toxicity measurement."
        },
        {
            "q": "How many pi bonds are generally present in a carbon–carbon triple bond?",
            "options": ["Zero", "One", "Two", "Three"],
            "answer": "Two",
            "explain": "A triple bond contains one sigma bond and two pi bonds."
        },
    ]
    if st.button("Start / restart quiz"):
        st.session_state.quiz_index = 0
        st.session_state.quiz_score = 0
        st.session_state.quiz_answered = False
        st.session_state.pop("quiz_choice", None)

    idx = st.session_state.quiz_index
    if idx < len(quiz):
        item = quiz[idx]
        st.write(f"**Question {idx + 1} of {len(quiz)}**")
        st.write(item["q"])
        choice = st.radio(
            "Choose one answer",
            item["options"],
            key=f"quiz_choice_{idx}",
            disabled=st.session_state.quiz_answered
        )
        if not st.session_state.quiz_answered:
            if st.button("Check answer"):
                st.session_state.quiz_answered = True
                if choice == item["answer"]:
                    st.session_state.quiz_score += 1
                st.rerun()
        else:
            if choice == item["answer"]:
                st.success("Correct!")
            else:
                st.warning(f"Not quite. Correct answer: {item['answer']}")
            st.info(item["explain"])
            if st.button("Next question"):
                st.session_state.quiz_index += 1
                st.session_state.quiz_answered = False
                st.rerun()
    else:
        st.success(f"Quiz complete! Score: {st.session_state.quiz_score}/{len(quiz)}")
        if st.button("Try quiz again"):
            st.session_state.quiz_index = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_answered = False
            st.rerun()

st.divider()
st.caption(
    "Educational tool only. Molecular descriptors are model-derived estimates and should "
    "not be used alone for medical, toxicological, or clinical decisions."
)
