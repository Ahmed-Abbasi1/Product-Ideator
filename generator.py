import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq + LangChain
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192"
)

def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def generate_response(domain: str, prompt_path: str) -> str:
    prompt_template = load_prompt(prompt_path)
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content=prompt_template + f"\n\nDomain/Idea:\n{domain}")
    ]
    response = llm(messages)
    return response.content.strip()
