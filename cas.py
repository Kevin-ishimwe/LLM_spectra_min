import cirpy
import logging

def generate_smiles_string(cas_number):
    try:
        print(cirpy.resolve(cas_number, 'smiles'))
    except Exception as e:
        logging.error(str(e))
        return "FAILED"


# http://cactus.nci.nih.gov/chemical/structure/acetamide/smiles
if (__name__=="__main__"):
 try:
    generate_smiles_string("acetamide")
 except Exception as e:
    logging.error(str(e))

