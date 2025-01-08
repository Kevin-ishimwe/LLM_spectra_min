import os
from dotenv import load_dotenv
#models 
#models 
from openai import OpenAI
import google.generativeai as genai
#prompts and helpers
import google.generativeai as genai
#prompts and helpers
from prompts import system_prompt
from token_counter import count_tokens
# loading variables from .env file
load_dotenv()
#safely handling my api keys
OAI_KEY=os.environ["OPEN_AI_KEY"]
GEMINI_KEY=os.environ["OPEN_AI_KEY"]


# model configs 
client = OpenAI(api_key=OAI_KEY)
genai.configure(api_key=GEMINI_KEY)


#safely handling my api keys
OAI_KEY=os.environ["OPEN_AI_KEY"]
GEMINI_KEY=os.environ["GEMINI_AI_KEY"]


# model configs 
client = OpenAI(api_key=OAI_KEY)
genai.configure(api_key=GEMINI_KEY)


#use openai gtp4o for inference 
def call_openAI(conversation,prompt, model = "gpt-4o",temperature=1):
  print(model)
  conversation.append( {"role": "user", "content":prompt })
  completion = client.chat.completions.create(
  temperature=temperature,
  model=model,
  messages=conversation)
  txt=completion.choices[0].message.content
  conversation.append({"role": "assistant", "content": txt})
  return txt,txt.split("### Start answer ###")[1].split("### End answer ###")[0].replace('\n','').strip().lower()


# Llama 3.2 90B
def call_llama(conversation,prompt, model = "gpt-4o",temperature=1):
  pass

  
# gemini
def call_gemini(conversation,prompt, model = "gemini-2.0-flash-exp",temperature=1):
  conversation.append( {"role": "user", "parts":prompt })
  genai.configure(api_key=GEMINI_KEY)
  model = genai.GenerativeModel(model,system_instruction=system_prompt)
  response = model.generate_content(prompt)
  txt=response.text
  conversation.append({"role": "model", "parts": txt})
  return txt,txt.split("### Start answer ###")[1].split("### End answer ###")[0].replace('\n','').strip().lower()
