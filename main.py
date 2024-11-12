
import os
from prompts import system_prompt,user_prompt
from extractor import json_extractor,csv_extractor
from llms import call_openAI
from crosscheck import cross_check_molecule,normalize_string
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def write_benchmark_result(file,reference,prediction,verdict):
    csv_file=open(file,"a")
    csv_file.write('\n'+reference+','+prediction+','+verdict)

# cross reference the prediction
def cross_check(molecules,challenge,prediction=""):
    for molecule in molecules:
        if(molecule[0]==(challenge['nmr_challenge_id']) and molecule[1]==challenge['formula'].lower()):
            if(normalize_string(molecule[2])==normalize_string(prediction) or cross_check_molecule(prediction, molecule[2])):
                logging.info("\033[94m \nMOLECULE PREDICTION TRUE \x1b[0m")
                reference=str(molecule[0])+','+challenge['formula']+','+molecule[2]
                write_benchmark_result("./NMR_Set/easy/Benchmark-result-EASY-1.csv",reference,prediction,'1')
                return True
            else:
                 logging.info("\033[91m \nMOLECULE PREDICTION FALSE \x1b[0m")
                 reference=str(molecule[0])+','+challenge['formula']+','+molecule[2]
                 write_benchmark_result("./NMR_Set/easy/Benchmark-result-EASY-1.csv",reference,prediction,'0')
                 return False
            break

def run_batch(file):
    molecules= json_extractor(file)
    for index,molecule in enumerate (molecules):
        run_single(molecule)


def run_single(molecule): 
    correct_reference=csv_extractor("./NMR_Set/easy/nmr_challenge_ids_easy.csv")
    response,model_prediction=call_openAI(molecule)
    cross_check(correct_reference,molecule,model_prediction)



if (__name__=="__main__"):
 try:
    run_batch('./NMR_Set/easy/NMRspectra_EASY.json')

 except Exception as e:
        print("[ERROR]\n",str(e))