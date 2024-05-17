import os
from io import BytesIO
import requests
from dotenv import load_dotenv, find_dotenv
import fitz
from langchain_community.vectorstores import AstraDB
from langchain.retrievers.self_query.astradb import AstraDBTranslator
from langchain_openai import OpenAI, ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity

##################
# S E T T I N G S
##################

#Loading variables of the environment
found_dotenv = find_dotenv()
load_dotenv(found_dotenv)
the_key = os.getenv("BU_OPENAI_API_KEY")
astradb_key = os.getenv("ASTRADB_TOKEN")
astradb_api_endpoint = os.getenv("ASTRADB_API_ENDPOINT")
pdf_target_url = "https://api.akabot-staging.zetta-demo.space/fileuploads/Artificial-Intelligence-in-Finance-6a364d95-f26c-41e6-a3a1-54f9b9f975d2.pdf"


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

####################
# F U N C T I O N S
####################


def load_pdf():
    response = requests.get(pdf_target_url)
    if response.status_code == 200:
        myfile = BytesIO(response.content)
    doc = fitz.open(stream=myfile, filetype="pdf")
    print(f"Number of pages: {doc.page_count}")
    return doc

def extract_text_from_pdf(doc):
    texts = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text").replace("\n", "")
        texts.append({
            "page_number": page_num + 1,  
            "page_content": text
        })
    return texts

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=128,
        length_function=len,
        add_start_index=True
    )
    chunks_with_metadata = []
    for document in documents:
        chunks = text_splitter.split_text(document["page_content"])
        for i, chunk in enumerate(chunks):
            chunks_with_metadata.append({
                "chunk_id": f"{document['page_number']}_{i}",
                "content": chunk,
                "page_number": document['page_number'],
            })

    print(f"Split {len(documents)} documents into {len(chunks_with_metadata)} chunks.")
    return chunks_with_metadata

def custom_retriever(query, embedding_model_instance, vstore):
    
    results = vstore.similarity_search(query, k=3)
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")

    return results

################
# P R O C E S S
################

doc = load_pdf()

documents = extract_text_from_pdf(doc)

splitted_text = split_text(documents)


for i, chunk in enumerate(splitted_text[:5]):
    print(f"\nChunk {i}: {chunk}\n")


documents_to_store = [Document(page_content=chunk["content"], metadata={"chunk_id": chunk["chunk_id"], "page_number": chunk["page_number"]}) for chunk in splitted_text]

# vstore.add_documents(documents_to_store) # To use once

query = "What are the applications of AI in finance?"
results = custom_retriever(query, embedding_model_instance, vstore)



# vstore.clear() #To delete all the vectors of the collection, so to use once

##############
# R E S U L T 
##############

# Display the results
print("\n\n-Here is the result-\n\n")
for result in results:
    print(f"\n\nChunk ID: {result.metadata['chunk_id']}")
    print(f"Page Number: {result.metadata['page_number']}")
    print(f"Content: {result.page_content}\n")

