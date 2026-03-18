import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

# optional reranker
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

@st.cache_resource
def build_vectorstore(corpus_dir="./corpus", chunk_size=600, chunk_overlap=100, embedding_model="all-MiniLM-L6-v2"):
    loader = DirectoryLoader(corpus_dir, glob="./*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def rerank(question, docs, reranker_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    if CrossEncoder is None:
        st.warning("`sentence-transformers` non disponible — impossible de reranker.")
        return docs
    reranker = CrossEncoder(reranker_name)
    pairs = [(question, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for s, d in scored]


def generate_answer(question, docs):
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""Tu es un expert de One Piece spécialisé dans l'arc Marineford. Utilise les extraits suivants pour répondre à la question. Si tu ne sais pas, dis que tu ne sais pas.\n\nCONTEXTE :\n{context}\n\nQUESTION :\n{question}\n\nRÉPONSE (en Markdown) :"""
    resp = client.chat.completions.create(model="google/gemini-2.0-flash-001", messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content


st.set_page_config(page_title="Nina Goislot - One Piece RAG", layout="wide")
st.title("⚔️ One Piece RAG - Nina Goislot")

with st.sidebar:
    st.header("Index / RAG settings")
    corpus_dir = st.text_input("Corpus folder", value="./corpus")
    chunk_size = st.number_input("Chunk size", value=600, step=100)
    chunk_overlap = st.number_input("Chunk overlap", value=100, step=50)
    embedding_model = st.text_input("Embedding model", value="all-MiniLM-L6-v2")
    k = st.number_input("Top-K to return", min_value=1, max_value=20, value=3)
    st.markdown("---")
    st.header("Reranking")
    enable_rerank = st.checkbox("Enable reranking (CrossEncoder)", value=False)
    reranker_model = st.text_input("Reranker model", value="cross-encoder/ms-marco-MiniLM-L-6-v2")
    st.markdown("---")
    st.header("Evaluation")
    use_llm_judge = st.checkbox("Use LLM-as-judge for scoring", value=True)
    st.markdown("(LLM-as-judge will consume API quota)")

# build or load index
with st.spinner("Building / loading vectorstore — patience please..."):
    try:
        vs = build_vectorstore(corpus_dir=corpus_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap, embedding_model=embedding_model)
    except Exception as e:
        st.error(f"Impossible de construire le vectorstore: {e}")
        st.stop()

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Question & Answer (en anglais de préférence)")
    question = st.text_area("Pose ta question mon goat", height=160)
    if st.button("Envoyer"):
        if not question.strip():
            st.warning("Écris une question d'abord >:(")
        else:
            with st.spinner("Recherche + génération..."):
                docs = vs.similarity_search(question, k=max(10, k))
                if enable_rerank:
                    docs = rerank(question, docs, reranker_name=reranker_model)
                top_docs = docs[:k]
                answer = generate_answer(question, top_docs)
            st.markdown("**Réponse :**")
            st.markdown(answer)
            st.markdown("**Chunks retournés :**")
            for i, d in enumerate(top_docs, 1):
                src = getattr(d, 'metadata', {}).get('source', 'unknown')
                st.markdown(f"**Chunk {i}** — _source: {src}_")
                st.write(d.page_content[:800])

with col2:
    st.subheader("Evaluation rapide")
    st.write("Format: une paire par ligne, séparateur `||` → question||expected_answer")
    test_input = st.text_area("Pastes tes paires ici", value="Who killed Ace?||Akainu killed Ace.\nWhat is Gura Gura no Mi?||A Devil Fruit that creates powerful shockwaves.")
    if st.button("Lancer l'évaluation"):
        lines = [l.strip() for l in test_input.splitlines() if l.strip()]
        pairs = []
        for ln in lines:
            if '||' in ln:
                q, ea = ln.split('||', 1)
                pairs.append((q.strip(), ea.strip()))
        if not pairs:
            st.warning("Aucune paire valide trouvée.")
        else:
            results = []
            with st.spinner("Évaluation en cours..."):
                for (q, ea) in pairs:
                    initial_docs = vs.similarity_search(q, k=10)
                    top_k_docs = initial_docs[:k]
                    relevant_counts = sum(1 for d in top_k_docs if ea.lower() in d.page_content.lower())
                    precision_at_k = relevant_counts / max(1, k)
                    recall_at_k = 1.0 if any(ea.lower() in d.page_content.lower() for d in initial_docs) else 0.0
                    # generation
                    if enable_rerank:
                        gen_docs = rerank(q, initial_docs, reranker_name=reranker_model)[:k]
                    else:
                        gen_docs = top_k_docs
                    rag_answer = generate_answer(q, gen_docs)
                    judge_out = None
                    if use_llm_judge:
                        judge_prompt = f"Compare ces deux réponses et donne une note de 1 à 5 :\n\nQuestion : {q}\nRéponse attendue : {ea}\nRéponse du RAG : {rag_answer}\n\nNote (1=hors sujet, 5=parfait) et justification :"
                        try:
                            resp = client.chat.completions.create(model="google/gemini-2.0-flash-001", messages=[{"role": "user", "content": judge_prompt}])
                            judge_out = resp.choices[0].message.content
                        except Exception as e:
                            judge_out = f"Erreur judge: {e}"
                    results.append({
                        'q': q,
                        'expected': ea,
                        'precision@k': precision_at_k,
                        'recall@k': recall_at_k,
                        'rag_answer': rag_answer,
                        'judge': judge_out,
                    })
            # aggregation
            avg_p = sum(r['precision@k'] for r in results) / len(results)
            avg_r = sum(r['recall@k'] for r in results) / len(results)
            st.markdown(f"**Résultats (n={len(results)})** — Precision@{k}: **{avg_p:.2f}**, Recall@{k}: **{avg_r:.2f}**")
            for r in results:
                st.markdown("---")
                st.markdown(f"**Q:** {r['q']}")
                st.markdown(f"**Expected:** {r['expected']}")
                st.markdown(f"**Precision@{k}:** {r['precision@k']:.2f} — **Recall@{k}:** {r['recall@k']:.1f}")
                st.markdown("**RAG answer:**")
                st.markdown(r['rag_answer'])
                if r['judge']:
                    st.markdown("**LLM-as-judge:**")
                    st.markdown(r['judge'])

st.markdown("---")
st.info("Lancer localement: `streamlit run streamlit_app.py`. Le reranking nécessite `sentence-transformers`.")
