from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdChemReactions
from IPython.display import display, SVG

from morganrxn.core.molecule_utils import get_mol_ecfp_atom_to_bits


def plot_mol(mol):
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
    display(mol)  # Displays the reaction visually in Jupyter


def plot_reaction(rxn):
    if type(rxn) == str:
        rxn = rdChemReactions.ReactionFromSmarts(rxn)
    display(rxn)  # Displays the reaction visually in Jupyter


def plot_mol_ecfp(mol, ecfp_params, img_size=600, font_size=15, use_svg=False, highlight_atoms=None, morgan_bits=[], put_indices=False):
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)

    # Compute 2D coordinates to ensure tight layout
    AllChem.Compute2DCoords(mol)

    # Compute Morgan bits associated with atoms
    atoms_morgan_bits = get_mol_ecfp_atom_to_bits(mol, ecfp_params=ecfp_params)

    # Drawing options
    opts = Draw.MolDrawOptions()
    opts.fixedBondLength = 50  # Adjust bond length for better spacing
    opts.maxFontSize = font_size  # Improve font readability
    opts.bondLineWidth = 2  # try 3–4 if you want thicker lines
    opts.padding = 0.0  # Remove extra padding to reduce white margins
    opts.useDefaultAtomPalette()  # Avoid extra spacing issues

    # Show atom numbers and Morgan bits
    for atom in mol.GetAtoms():
        i_rdkit = atom.GetIdx()
        if highlight_atoms is None or i_rdkit in highlight_atoms:
            symbol = atom.GetSymbol()
            if len(morgan_bits) == 0:
                #i_smarts = atom.GetAtomMapNum()
                #atom_label = f"{symbol}({i_rdkit},{i_smarts}):" + ",".join(map(str, atoms_morgan_bits.get(i_rdkit, [])))
                atom_label = f"{symbol}" 
                if put_indices:
                    atom_label = atom_label + f"{i_rdkit}"
                atom_label = atom_label + ":" + ",".join(map(str, atoms_morgan_bits.get(i_rdkit, [])))
                opts.atomLabels[i_rdkit] = atom_label
            else:
                atom_label = f"{symbol}"
                if put_indices:
                    atom_label = atom_label + f"{i_rdkit}"
                atom_label = atom_label + ":"
                atom_morgan_bits = atoms_morgan_bits.get(i_rdkit, [])
                if len([x for x in atom_morgan_bits if x in morgan_bits]) > 0:
                    for atom_morgan_bit in atom_morgan_bits:
                        if atom_morgan_bit in morgan_bits:
                            atom_label = atom_label + str(atom_morgan_bit) + ","
                        else:
                            atom_label = atom_label + "-,"
                atom_label = atom_label[:-1]
                opts.atomLabels[i_rdkit] = atom_label

    # Highlight specific atoms if provided
    highlight_colors = {idx: (1, 0, 0) for idx in highlight_atoms} if highlight_atoms else {}

    if use_svg:
        drawer = Draw.MolDraw2DSVG(img_size, img_size)
        drawer.SetDrawOptions(opts)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        display(SVG(svg))
    else:
        Draw.MolToImage(mol, size=(img_size, img_size), options=opts,
                               highlightAtoms=list(highlight_atoms) if highlight_atoms else None,
                               highlightAtomColors=highlight_colors if highlight_colors else None).show()
