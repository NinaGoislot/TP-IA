import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

# 1. Configuration & Clés API
load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def build_rag_system():
    print("--- Chargement du corpus Marineford ---")
    # Charge tous les fichiers .txt du dossier corpus
    loader = DirectoryLoader(
    './corpus', 
    glob="./*.txt", 
    loader_cls=TextLoader, 
    loader_kwargs={'encoding': 'utf-8'}  # <--- Ajoute ça !
)
    documents = loader.load()

    # 2. Chunking (Découpage)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Nombre de chunks créés : {len(chunks)}")

    # 3. Vectorisation (Embeddings)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Stockage (Base de données vectorielle)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def ask_question(question, vectorstore):
    # 5. Retrieval (Recherche des 3 chunks les plus proches)
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 6. Génération (Prompt + LLM)
    prompt = f"""Tu es un expert de One Piece spécialisé dans l'arc Marineford. 
Utilise les extraits suivants pour répondre à la question. 
Si tu ne sais pas, dis que tu ne sais pas.

CONTEXTE :
{context}

QUESTION : 
{question}

RÉPONSE (en Markdown) :"""

    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-001", # Modèle gratuit sur OpenRouter
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    # Initialisation
    vs = build_rag_system()
    
    # Test interactif
    print("\n--- RAG prêt ! Posez votre question sur Marineford (ou 'exit') ---")
    while True:
        query = input("\nVotre question : ")
        if query.lower() == 'exit': break
        
        answer = ask_question(query, vs)
        print(f"\nRéponse :\n{answer}")