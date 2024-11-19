import os
from dotenv import load_dotenv
from openai import OpenAI
from prompts import system_prompt,user_prompt
from token_counter import count_tokens
# loading variables from .env file
load_dotenv()
#safely handling my api key and working with the openai client
API_KEY=os.environ["OPEN_AI_KEY"]
client = OpenAI(
    api_key=API_KEY
  )
#use openai gtp4o for inference 
def call_openAI(conversation,prompt, model = "gpt-4o",temperature=1):
  conversation.append( {"role": "user", "content":prompt })
  completion = client.chat.completions.create(
  temperature=temperature,
  model=model,
  messages=conversation)
  txt=completion.choices[0].message.content
  conversation.append({"role": "assistant", "content": txt})
  return txt,txt.split("### Start answer ###")[1].split("### End answer ###")[0].replace('\n','').strip().lower()
