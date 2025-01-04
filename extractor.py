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
        return [({
            'id':id,
            'formula':formula.strip(),
            'name':name.lower().strip(),
            'smile':smile
        }) 
                for id,formula,name,smile in zip(molecules['NMR_Challenge_ID'],molecules['Formula'],molecules['True names'],molecules['Smiles'])]
    except Exception as e:
        print("[ERROR]\n",str(e))


def token_formater(data):
    peaks=""
    for peak in data["peaks"]:
        peaks+="--------\n"+"name: "+peak['name']+"\n"
        peaks+="Shift: "+str(peak['shift'])+"\n"
        peaks+="Range: "+(" ".join(str(value) for value in peak['range']))+"\n"
        peaks+="Hydrogens: "+str(peak['hydrogens'])+"\n"
        peaks+="Integral: "+str(peak['integral'])+"\n"
        peaks+="Class: "+str(peak['class'])+"\n"
        peaks+="J values: "+(" ".join(str(value) for value in peak['j_values']))+"\n"
        peaks+="Method: "+str(peak['method'])+"\n\n"
    
    return f"""\nFormula: {data['formula']}\nPeaks {peaks}
    """
if (__name__=="__main__"):
 try:
    #usecase: correct_data=csv_extractor("./NMR_Set/nmr_spectra_hard.json")
    # usecase:json
    NMR_data=json_extractor("test.json")
    print(token_formater(NMR_data))
    
 except Exception as e:
    print("[ERROR]\n",str(e))