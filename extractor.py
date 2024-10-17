import json

def json_extractor(filename):
    file = open(filename,"r")
    txt=file.read()
    json_data=json.loads(txt)
    return json_data


if (__name__=="__main__"):
 try:
    NMR_data=json_extractor("./NMR_Set/nmr_spectra_hard.json")
    for NMR_spectra in NMR_data:
        print (NMR_spectra)
 except Exception as e:
    print("[ERROR]\n",str(e))