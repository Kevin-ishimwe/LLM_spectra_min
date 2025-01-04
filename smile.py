import cirpy
import logging
import time

def generate_smiles_string(cas_number):
    try:
        #delay so i dont crash the servers
        time.sleep(15)
        return cirpy.resolve(cas_number, 'smiles')
    except Exception as e:
        logging.error(str(e))
        return "FAILED"

# http://cactus.nci.nih.gov/chemical/structure/acetamide/smiles
if (__name__=="__main__"):
 try:
    generate_smiles_string("1,4-dioxane-2,5-dione")
 except Exception as e:
    logging.error(str(e))

