import json
import pandas as pd

def json_extractor(filename):
 try:
    file = open(filename,"r")
    txt=file.read()
    json_data=json.loads(txt)
    return json_data
 except Exception as e:
    print("[ERROR]\n",str(e))

def csv_extractor(filename):
    try:
        file=pd.read_csv(filename)
        molecules=pd.DataFrame(file)
        return [(id,formula.lower().strip(),name.lower().strip()) for id,formula,name in zip(molecules['NMR_Challenge_ID'],molecules['Formula'],molecules['True names '])]
    except Exception as e:
        print("[ERROR]\n",str(e))



if (__name__=="__main__"):
 try:
    #usecase: correct_data=csv_extractor("./NMR_Set/nmr_spectra_hard.json")
    # usecase:json
    NMR_data=json_extractor("./NMR_Set/nmr_spectra_hard.json")
    for NMR_spectra in NMR_data:
        print (NMR_spectra)
 except Exception as e:
    print("[ERROR]\n",str(e))