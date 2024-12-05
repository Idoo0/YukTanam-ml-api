from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

def getLLM():  
  return ChatGoogleGenerativeAI(
      model="gemini-1.5-flash",
      google_api_key=os.getenv("GEMINI_API_KEY")
  )