from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
#only generates stereoisomers 
def generate_isomers(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    isomers = set()
    for perm in AllChem.EnumerateStereoisomers(mol):
        isomer_smiles = Chem.MolToSmiles(perm, isomericSmiles=True)
        isomers.add(isomer_smiles)
    
    return list(isomers)

# Example usage
smiles = "CC(O)C(O)=O"  # Lactic acid
isomers = generate_isomers(smiles)
for i, isomer in enumerate(isomers):
    print(f"Isomer {i+1}: {isomer}")
