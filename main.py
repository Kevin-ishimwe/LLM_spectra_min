
import os
from prompts import system_prompt,user_prompt,consistency_reprompt,regeneration_prompt
from extractor import json_extractor,csv_extractor
from llms import call_openAI
from crosscheck import cross_check_molecule,normalize_string
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def write_benchmark_result(file,reference,prediction,verdict):
    csv_file=open(file,"a")
    csv_file.write('\n'+f"{reference}"+','+f"\"{prediction}\""+','+verdict)

def prepare_output_file(path):
    try:
        csv_file=open(path,"a")
        csv_file.write("Id,Formula,TrueName,Prediction,Verdict")
    except Exception as e:
        logging.error(str(e))


# cross reference the prediction
def cross_check(references,challenge,prediction,outfile):
    for molecule in references:
        if(molecule[0]==(challenge['nmr_challenge_id']) and molecule[1]==challenge['formula'].lower()):
            if(normalize_string(molecule[2])==normalize_string(prediction) or cross_check_molecule(prediction, molecule[2])):
                logging.info("\033[32m \nMOLECULE PREDICTION TRUE \x1b[0m")
                reference=str(molecule[0])+','+challenge['formula']+','+f"\"{molecule[2]}\""
                write_benchmark_result(outfile,reference,prediction,'1')
                return True
            else:
                 logging.info("\033[91m \nMOLECULE PREDICTION FALSE \x1b[0m")
                 reference=str(molecule[0])+','+challenge['formula']+','+molecule[2]
                 write_benchmark_result(outfile,reference,prediction,'0')
                 return False        


def run_batch(temperature=0.8,id_file="N/A",challenge_file="N/A",outputfile="N/A"):
    correct_reference=csv_extractor(id_file)
    molecules= json_extractor(challenge_file)
    for index,molecule in enumerate (molecules):
        model_prediction=run_single(temperature,molecule)
        cross_check(references=correct_reference,
                    challenge=molecule,
                    prediction=model_prediction,
                    outfile=outputfile)


def run_single(temperature,molecule): 
    conversation=[]
    conversation.append
    ( {
        "role": "system", 
        "content":system_prompt
        })
    prompt=f"""{user_prompt(molecule)}"""
    response,model_prediction=call_openAI(conversation=conversation,prompt=prompt,temperature=temperature)
    # check molecular formula constency
    response,constitent=call_openAI(conversation=conversation,prompt=consistency_reprompt(),temperature=temperature)
    if (constitent.lower()=="yes"):
        logging.info("\033[34m \nMOLECULAR FORMULA CONSISTENT WITH PREDICTION \x1b[0m")
    else:
        logging.info("\033[41m \nMOLECULAR FORMULA NOT CONSISTENT WITH PREDICTION \nRE-PROMPTING ....\x1b[0m")
        response,model_prediction=call_openAI(conversation=conversation,prompt=regeneration_prompt(),temperature=temperature)
    return model_prediction

if (__name__=="__main__"):
 try:
    MODEL = "gpt-4o"
    TEMPERATURE=0
    CHALLENGE_FILE="./NMR_Set/easy/NMRspectra_EASY.json"
    CHALLENGE_IDS="./NMR_Set/easy/nmr_challenge_ids_easy.csv"
    SAMPLE_RUN="./NMR_Set/easy/sample_run.json"
    OUTPUTFILE="./NMR_Set/test/Benchmark-result-test-run.csv"
    prepare_output_file(OUTPUTFILE)
    run_batch(
        temperature=TEMPERATURE,
        id_file=CHALLENGE_IDS,
        challenge_file=CHALLENGE_FILE,
        outputfile=OUTPUTFILE
        )
 except Exception as e:
        print("[ERROR]\n",str(e))