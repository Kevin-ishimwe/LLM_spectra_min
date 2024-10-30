
from prompts import system_prompt,user_prompt
from extractor import json_extractor,csv_extractor
from llms import call_openAI
from crosscheck import cross_check_molecule,normalize_string
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# cross reference the prediction
def cross_check(molecules,challenge,prediction=""):
    for molecule in molecules:
        if(molecule[0]==(challenge['nmr_challenge_id']) and molecule[1]==challenge['formula'].lower()):
            norm_pred=prediction
            if(normalize_string(molecule[2])==normalize_string(norm_pred) or cross_check_molecule(prediction, molecule[2])):
                logging.info("\033[94m \nMOLECULE PREDICTION TRUE \x1b[0m")
                return True
            else:
                 logging.info("\033[91m \nMOLECULE PREDICTION FALSE \x1b[0m")
                 return False
            break


def run_batch(file):
    molecules= json_extractor(file)
    for molecule,index in enumerate (molecules):
        print(index)

    pass

def sample_run_single(): 
    file="./NMR_Set/easy/sample_run.json"
    sample= json_extractor(file)
    correct_reference=csv_extractor("./NMR_Set/easy/nmr_challenge_ids_easy.csv")
    response,model_prediction=call_openAI(sample)
    cross_check(correct_reference,sample,model_prediction)



if (__name__=="__main__"):
 try:
    #file ='./NMR_Set/NMR spectra_EASY.json'
    #run_batch(file)
    sample_run_single()
    #cross_check(correct_reference,"")

    

 except Exception as e:
        print("[ERROR]\n",str(e))