
import os
from dotenv import load_dotenv
import json
from openai import OpenAI
from prompts import system_prompt,user_prompt
from extractor import json_extractor

# loading variables from .env file
load_dotenv()
#safely handling my api key and working with the openai client
API_KEY=os.environ["OPEN_AI_KEY"]
client = OpenAI(
    api_key=API_KEY
  )
# cross reference the prediction
def cross_check(correct_name,prediction):
    return False

#use openai gtp4o for inference 
def call_openAI(NMR_data, model = "gpt-4o",system_prompt=system_prompt):
  completion = client.chat.completions.create(
  temperature=0.8,
  model=model,
  messages=[
    {"role": "system", "content":system_prompt} ,
    {"role": "user", "content": f"""{user_prompt(NMR_data)}"""}
  ]
  )
  txt=completion.choices[0].message.content
  print(txt)
  return txt


if (__name__=="__main__"):
 try:
    file="./NMR_Set/sample_run.json"
    sample= json_extractor(file)
    call_openAI(sample)

 except Exception as e:
        print("[ERROR]\n",str(e))