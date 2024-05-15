import os
from io import BytesIO
import requests
from dotenv import load_dotenv, find_dotenv
import fitz
from langchain_community.vectorstores import AstraDB
from langchain_openai import OpenAI, ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


found_dotenv = find_dotenv()
load_dotenv(found_dotenv)
the_key = os.getenv("BU_OPENAI_API_KEY")
astradb_key = os.getenv("ASTRADB_TOKEN")
astradb_api_endpoint = os.getenv("ASTRADB_API_ENDPOINT")
pdf_target_url = "https://api.akabot-staging.zetta-demo.space/fileuploads/Artificial-Intelligence-in-Finance-6a364d95-f26c-41e6-a3a1-54f9b9f975d2.pdf"


# Debug statements to verify environment variables
print(f"API Key: {the_key}")
print(f"AstraDB Token: {astradb_key}")
print(f"AstraDB API Endpoint: {astradb_api_endpoint}")

# Models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo-0125"
llm = ChatOpenAI(api_key=the_key, model=GPT_MODEL)
embedding_model_instance = OpenAIEmbeddings(api_key=the_key, model=EMBEDDING_MODEL)


# vstore = AstraDB(
#     embedding=embedding_model_instance,
#     collection_name="astra_vector_introduction",
#     api_endpoint=astradb_api_endpoint,
#     token=astradb_key
# )

def load_pdf():
    response = requests.get(pdf_target_url)
    if response.status_code == 200:
        myfile = BytesIO(response.content)
    text = ""
    doc = fitz.open(stream=myfile, filetype="pdf")
    print(f"Number of page : {doc.page_count}")
    return doc

def extract_text_from_pdf(doc):
    texts = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        texts.append(text)
    return texts

def split_text(texts):
    print(texts[0])
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_text(texts[0])
    print(f"Split {len(texts)} documents into {len(chunks)} chunks.")
    print("\n\n- Chunks - \n")
    for chunk in chunks:
        print(chunks)
    return chunks


doc = load_pdf()
texts = extract_text_from_pdf(doc)
splitted_text = split_text(texts)

