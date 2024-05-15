import os
from dotenv import load_dotenv, find_dotenv
import fitz
from langchain_community.vectorstores import AstraDB
from langchain_openai import OpenAI, ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings


found_dotenv = find_dotenv()
load_dotenv(found_dotenv)
the_key = os.getenv("BU_OPENAI_API_KEY")
astradb_key = os.getenv("ASTRADB_TOKEN")
astradb_api_endpoint = os.getenv("ASTRADB_API_ENDPOINT")

# Debug statements to verify environment variables
print(f"API Key: {the_key}")
print(f"AstraDB Token: {astradb_key}")
print(f"AstraDB API Endpoint: {astradb_api_endpoint}")

# Models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo-0125"
llm = ChatOpenAI(api_key=the_key, model=GPT_MODEL)
embedding_model_instance = OpenAIEmbeddings(api_key=the_key, model=EMBEDDING_MODEL)


vstore = AstraDB(
    embedding=embedding_model_instance,
    collection_name="astra_vector_introduction",
    api_endpoint=astradb_api_endpoint,
    token=astradb_key
)

def load_pdf():
    doc = fitz.open("NLP.pdf")
    print(f"Number of page : {doc.page_count}")
    return doc

load_pdf()


