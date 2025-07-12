# --- app_bert.py (Streamlit Web App for Italian BERT - LOGO GRANDE CENTRALE TRAMITE ST.IMAGE) ---

import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
import os
from huggingface_hub import hf_hub_download

# --- Interfaccia Utente di Streamlit (Configurazione, DEVE ESSERE LA PRIMA COSA!) ---
st.set_page_config(
    page_title="PoisonChat",
    layout="centered",
    page_icon="poisonchatshifted.png", # Il tuo favicon
    initial_sidebar_state="collapsed"
)

# --- Configurazione ---
HF_MODEL_REPO = "AngeloTetro/PoisonChat"
HF_SUBFOLDER_NAME = "bert_italian_category_webapp_model"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Caricamento del Modello, Tokenizer e Label Encoder ---
@st.cache_resource
def load_model_and_tokenizer():
    with st.spinner("Caricamento modello PoisonChat..."):
        try:
            tokenizer = BertTokenizer.from_pretrained(HF_MODEL_REPO, subfolder=HF_SUBFOLDER_NAME)
            model = BertForSequenceClassification.from_pretrained(HF_MODEL_REPO, subfolder=HF_SUBFOLDER_NAME).to(DEVICE)
            model.eval()

            label_encoder_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="label_encoder.joblib", subfolder=HF_SUBFOLDER_NAME)
            label_encoder = joblib.load(label_encoder_path)

            return tokenizer, model, label_encoder
        except Exception as e:
            st.error(f"Errore critico durante il caricamento del modello: {e}")
            st.info(f"Controlla il repository {HF_MODEL_REPO} e la sottocartella {HF_SUBFOLDER_NAME} su Hugging Face Hub.")
            st.stop()

tokenizer, model, label_encoder = load_model_and_tokenizer()

# --- Funzione di Predizione ---
def predict_category(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)

    predicted_id = torch.argmax(probabilities, dim=1).item()
    predicted_category = label_encoder.inverse_transform([predicted_id])[0]
    predicted_probability = probabilities[0][predicted_id].item()

    all_probabilities = {
        label_encoder.inverse_transform([i])[0]: prob.item()
        for i, prob in enumerate(probabilities[0])
    }

    return predicted_category, predicted_probability, all_probabilities

# --- Interfaccia Utente di Streamlit ---

# Centra il logo grande con st.image e colonne
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("poisonchatbetter.png", width=250) # Torna a st.image, path diretto
    
# Titolo principale centrato
st.markdown("<h1 style='text-align: center; color: white;'>PoisonChat</h1>", unsafe_allow_html=True)

# Sottotitolo modificato e centrato
st.markdown("<h3 style='text-align: center; color: #ADD8E6;'>Classificatore di Conversazioni</h1>", unsafe_allow_html=True)

# Testo descrittivo centrato
st.markdown("""
<p style='text-align: center; font-size: 1.1em;'>
Questa applicazione classifica il testo di una conversazione in una delle categorie predefinite, utilizzando un modello BERT Italiano addestrato. Aiuta a identificare la natura delle interazioni.
</p>
""", unsafe_allow_html=True)

st.write("---") # Una linea separatrice

st.subheader("Inserisci il testo della conversazione:")
# Rimuovi l'etichetta dell'area di testo, dato che c'è già un subheader
user_input = st.text_area("", height=150, placeholder="Es: Ciao, come stai? Vorrei parlare di come risolvere la nostra discussione di ieri.")

if st.button("Classifica Categoria", use_container_width=True, type="primary"):
    if user_input:
        with st.spinner("Classificazione in corso..."):
            predicted_category, predicted_probability, all_probs = predict_category(user_input)

            st.success(f"**Categoria Predetta:** {predicted_category}")
            st.write(f"**Confidenza:** {predicted_probability:.2%}")

            with st.expander("Mostra Dettaglio delle Probabilità"):
                sorted_probs = sorted(all_probs.items(), key=lambda item: item[1], reverse=True)
                for category, prob in sorted_probs:
                    st.write(f"- {category}: {prob:.2%}")
    else:
        st.warning("Per favore, inserisci del testo per la classificazione.")

st.write("---")
st.info("Sviluppato con Streamlit e Hugging Face Transformers.")
