# RAG One Piece

## Description
Ce projet implémente un système de Retrieval-Augmented Generation (RAG) basé sur un corpus One Piece.

## Installation

```bash
pip install -r requierements.txt
```

## Lancer l'interface web

1. Créez un fichier `.env` à la racine du projet contenant votre clé OpenRouter :

```env
OPENROUTER_API_KEY=[your_key_here]
```

2. Lancer l'application Streamlit (recommandé) :

```bash
streamlit run streamlit_app.py
```
or, if you have a problem with the PATH, try this run :

```bash
python -m streamlit run streamlit_app.py
```

## Notes
- Le reranking utilise `sentence-transformers` et `CrossEncoder` (lent à charger).
- `LLM-as-judge` consomme des appels API.
