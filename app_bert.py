# --- app_bert.py (Streamlit Web App for Italian BERT - CON LOGO E FAVICON) ---

import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
import os
from huggingface_hub import hf_hub_download

# --- Interfaccia Utente di Streamlit (Configurazione, DEVE ESSERE LA PRIMA COSA!) ---
# Imposta il favicon (l'icona nella scheda del browser)
st.set_page_config(page_title="PoisonChat", layout="centered", page_icon="poisonchat.png")

# --- Configurazione ---
HF_MODEL_REPO = "AngeloTetro/PoisonChat"
HF_SUBFOLDER_NAME = "bert_italian_category_webapp_model" 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Caricamento del Modello, Tokenizer e Label Encoder ---
@st.cache_resource
def load_model_and_tokenizer():
    try:
        tokenizer = BertTokenizer.from_pretrained(HF_MODEL_REPO, subfolder=HF_SUBFOLDER_NAME)
        model = BertForSequenceClassification.from_pretrained(HF_MODEL_REPO, subfolder=HF_SUBFOLDER_NAME).to(DEVICE)
        model.eval()

        label_encoder_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="label_encoder.joblib", subfolder=HF_SUBFOLDER_NAME)
        label_encoder = joblib.load(label_encoder_path)

        st.success(f"Modello BERT, tokenizer e codificatore di etichette caricati da Hugging Face Hub: {HF_MODEL_REPO}/{HF_SUBFOLDER_NAME}!")
        return tokenizer, model, label_encoder
    except Exception as e:
        st.error(f"Errore durante il caricamento del modello da Hugging Face Hub: {e}")
        st.info(f"Dettaglio dell'errore: {e}")
        st.info(f"Assicurati che il repository '{HF_MODEL_REPO}' e la sottocartella '{HF_SUBFOLDER_NAME}' siano corretti e che tutti i file del modello siano stati caricati al loro interno.")
        st.stop()

tokenizer, model, label_encoder = load_model_and_tokenizer()

# --- Aggiungi l'immagine del logo all'inizio dell'app ---
# Assicurati che "poisonchat.png" sia nella stessa cartella di app_bert.py su GitHub
st.image("poisonchat.png", width=150) # Puoi regolare la larghezza del logo qui

# --- Interfaccia Utente di Streamlit ---

# Titolo principale
st.header("PoisonChat: Classificatore di Categorie di Conversazione") 

# Testo descrittivo senza collegamenti
st.markdown("""
Questa applicazione classifica il testo di una conversazione in una delle categorie predefinite, utilizzando un modello BERT Italiano addestrato. Aiuta a identificare la natura delle interazioni.
""")

st.subheader("Inserisci il testo della conversazione:")
user_input = st.text_area("Testo della conversazione:", height=150, placeholder="Es: Ciao, come stai? Vorrei parlare di come risolvere la nostra discussione di ieri.")

if st.button("Classifica Categoria"):
    if user_input:
        with st.spinner("Classificazione in corso..."):
            predicted_category, predicted_probability, all_probs = predict_category(user_input)
            
            st.success(f"**Categoria Predetta:** {predicted_category}")
            st.write(f"**Confidenza:** {predicted_probability:.2%}")

            st.subheader("Dettaglio delle Probabilità:")
            sorted_probs = sorted(all_probs.items(), key=lambda item: item[1], reverse=True)
            for category, prob in sorted_probs:
                st.write(f"- {category}: {prob:.2%}")
    else:
        st.warning("Per favore, inserisci del testo per la classificazione.")

st.markdown("---")
# Testo del footer senza collegamenti
st.info("Sviluppato con Streamlit e Hugging Face Transformers per PoisonChat.")
