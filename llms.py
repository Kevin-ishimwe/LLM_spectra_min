import os
from dotenv import load_dotenv
#models 
from anthropic import Anthropic
from openai import OpenAI
import google.generativeai as genai
#prompts and helpers

from prompts import system_prompt
from token_counter import count_tokens
# loading variables from .env file
load_dotenv()
#safely handling my api keys
OAI_KEY=os.environ["OPEN_AI_KEY"]
GEMINI_KEY=os.environ["GEMINI_AI_KEY"]
ANTHROPIC_KEY=os.environ["ANTHROPIC_API_KEY"]




#use openai gtp4o for inference 
def call_openAI(conversation,prompt, model = "gpt-4o",temperature=1):
  try:
    client = OpenAI(api_key=OAI_KEY)
    conversation.append( {"role": "user", "content":prompt })
    completion = client.chat.completions.create(
    temperature=temperature,
    model=model,
    messages=conversation)
    txt=completion.choices[0].message.content
    conversation.append({"role": "assistant", "content": txt})
    return txt,txt.split("### Start answer ###")[1].split("### End answer ###")[0].replace('\n','').strip().lower()
  except Exception as e:
    print("[ERROR] - openAI function:\n", str(e))
    raise


def call_claude_sonnet(conversation, prompt, temperature, model="claude-3-5-sonnet-20241022"):
    try:
        client = Anthropic(api_key=ANTHROPIC_KEY)
        conversation.append({"role": "user", "content": prompt})
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=temperature,
            system=system_prompt,
            messages=conversation
        )
        
        # The content is now accessed from response.content[0].text
        txt = response.content[0].text
        conversation.append({"role": "assistant", "content": txt})
        print(txt)
        return txt, txt.split("### Start answer ###")[1].split("### End answer ###")[0].replace('\n','').strip().lower()
        
    except Exception as e:
        print("[ERROR] - claude function:\n", str(e))
        raise
# gemini
def call_gemini(conversation,prompt, model = "gemini-2.0-flash-exp",temperature=1):
  genai.configure(api_key=GEMINI_KEY)
  conversation.append( {"role": "user", "parts":prompt })
  genai.configure(api_key=GEMINI_KEY)
  model = genai.GenerativeModel(model,system_instruction=system_prompt)
  response = model.generate_content(prompt)
  txt=response.text
  conversation.append({"role": "model", "parts": txt})
  return txt,txt.split("### Start answer ###")[1].split("### End answer ###")[0].replace('\n','').strip().lower()
