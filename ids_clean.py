import os
from extractor import csv_extractor,json_extractor
import random
import json
import shutil

ID_TRACKER=[]

def generate_new_ids(task_file,id_file,path_json,path_csv):
    tasks=json_extractor(task_file)
    ids=ids=csv_extractor (id_file)
    json_file=open(path_json,"w")
    csv_file=open(path_csv,"a")
    csv_file.write("NMR_Challenge_ID,Formula,True names,Smiles\n")
    json_task=[]
    for index,task in enumerate(tasks):
        if(task['nmr_challenge_id']==ids[index]['id']):
            random_id=shiftid_function(task['nmr_challenge_id'])
            while(random_id in ID_TRACKER):
                random_id=shiftid_function(task['nmr_challenge_id'])
            ID_TRACKER.append(random_id)
            try:
                task['nmr_challenge_id']=random_id
                json_task.append(task)
                #write csv file
                csv_string = f"{random_id},{ids[index]['formula']},\"{ids[index]['name']}\",{ids[index]['smile']}\n"
                csv_file.write(csv_string)
            except Exception as e:
                print("file write failed: ",str(e))
    json_file.write(json.dumps(json_task,indent=2))
    print('DONE GENRATING NEW ID DATASETS')

       
def shiftid_function(id):
    return random.randint(0, 118)+id
# so we dont accidental generate new ids
if(__name__=="____"):
    NEW_PATH="./NMR Datasets ID SHIFT/"
    IDS_ROOT="./H1 NMR Datasets/IDS/"
    CHALLENGE_ROOT="./H1 NMR Datasets/CHALLENGES/"

    TASK_FILES=[
        "nmr_spectra_easy.json",  
        "nmr_spectra_medium.json",
        "nmr_spectra_hard.json",
        ]
    IDS_FILES = [
        f"nmr_challenge_ids_easy.csv",
        f"nmr_challenge_med_ids.csv",
        f"nmr_challenge_har_ids.csv",
    ]
    # if the NEW_PATH exists. If so, delete it.
    if os.path.exists(NEW_PATH):
        shutil.rmtree(NEW_PATH)  

    # Recreate the NEW_PATH folder.
    os.makedirs(NEW_PATH)

    for i,TASK_FILE in enumerate(TASK_FILES):
        
        generate_new_ids(CHALLENGE_ROOT+TASK_FILE,# challenge task path
                        IDS_ROOT+IDS_FILES[i],  # id path
                        NEW_PATH+TASK_FILE, #n ew challenge task path
                        NEW_PATH+IDS_FILES[i] #new id task path
                        )
    