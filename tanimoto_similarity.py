#pip install rdkit
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs

def calculate_tanimoto(smile1, smile2):
  
        mol1 = Chem.MolFromSmiles(smile1)
        mol2 = Chem.MolFromSmiles(smile2)
        
        fp1 = rdFingerprintGenerator.GetFPs([mol1])
        fp2 = rdFingerprintGenerator.GetFPs([mol2])
        tanimoto_similarity = DataStructs.TanimotoSimilarity(fp1[0],fp2[0])
  
        return tanimoto_similarity

#example:
#calculate_tanimoto("CCC(C)O","CC(C)CO")
