# --- app_bert.py (Streamlit Web App for Italian BERT) ---

import streamlit as st
import torch
# IMPORTANTE: Usiamo BertTokenizer e BertForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
import os
from huggingface_hub import hf_hub_download

# --- Interfaccia Utente di Streamlit (Configurazione, DEVE ESSERE LA PRIMA COSA!) ---
st.set_page_config(page_title="PoisonChat", layout="centered")

# --- Configurazione ---
# Questo è il tuo repository esatto su Hugging Face Hub
HF_MODEL_REPO = "AngeloTetro/PoisonChat"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Caricamento del Modello, Tokenizer e Label Encoder ---
@st.cache_resource # Memorizza in cache il modello per caricarlo una sola volta
def load_model_and_tokenizer():
    try:
        # Carica tokenizer e modello ESPLICITAMENTE come BERT
        tokenizer = BertTokenizer.from_pretrained(HF_MODEL_REPO)
        model = BertForSequenceClassification.from_pretrained(HF_MODEL_REPO).to(DEVICE)
        model.eval() # Imposta il modello in modalità valutazione

        # Scarica il label_encoder.joblib da Hugging Face Hub
        label_encoder_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="label_encoder.joblib")
        label_encoder = joblib.load(label_encoder_path)

        st.success(f"Modello BERT, tokenizer e codificatore di etichette caricati da Hugging Face Hub: {HF_MODEL_REPO}!")
        return tokenizer, model, label_encoder
    except Exception as e:
        st.error(f"Errore durante il caricamento del modello da Hugging Face Hub: {e}")
        st.info(f"Dettaglio dell'errore: {e}") # Mostra l'errore completo per debug
        st.info(f"Assicurati che il repository {HF_MODEL_REPO} sia corretto e che tutti i file del modello (inclusi config.json, model.safetensors/pytorch_model.bin, vocab.txt, tokenizer.json, label_encoder.joblib) siano stati caricati correttamente su Hugging Face Hub.")
        st.stop() # Ferma l'esecuzione dello script se il caricamento fallisce

# Inizializza il modello, tokenizer e label_encoder al primo caricamento dell'app
tokenizer, model, label_encoder = load_model_and_tokenizer()

# --- Funzione di Predizione ---
def predict_category(text):
    # Tokenizza l'input e lo sposta sul dispositivo (CPU/GPU)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
    
    with torch.no_grad(): # Disabilita il calcolo dei gradienti per velocizzare l'inferenza
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1) # Converte i logits in probabilità

    # Ottieni la categoria predetta con la probabilità più alta
    predicted_id = torch.argmax(probabilities, dim=1).item()
    predicted_category = label_encoder.inverse_transform([predicted_id])[0]
    predicted_probability = probabilities[0][predicted_id].item()

    # Ottieni le probabilità per tutte le categorie
    all_probabilities = {
        label_encoder.inverse_transform([i])[0]: prob.item()
        for i, prob in enumerate(probabilities[0])
    }
    
    return predicted_category, predicted_probability, all_probabilities

# --- Interfaccia Utente di Streamlit ---

st.title("🐍 PoisonChat: Classificatore di Categorie di Conversazione") # Titolo visibile nell'app
st.markdown("""
Questa applicazione classifica il testo di una conversazione in una delle categorie predefinite, utilizzando un modello **BERT Italiano (dbmdz/bert-base-italian-uncased)** addestrato. Aiuta a identificare la natura delle interazioni.
""")

st.subheader("Inserisci il testo della conversazione:")
# Area di testo per l'input dell'utente
user_input = st.text_area("Testo della conversazione:", height=150, placeholder="Es: Ciao, come stai? Vorrei parlare di come risolvere la nostra discussione di ieri.")

# Pulsante per avviare la classificazione
if st.button("Classifica Categoria"):
    if user_input:
        with st.spinner("Classificazione in corso..."): # Mostra uno spinner mentre elabora
            predicted_category, predicted_probability, all_probs = predict_category(user_input)
            
            st.success(f"**Categoria Predetta:** {predicted_category}")
            st.write(f"**Confidenza:** {predicted_probability:.2%}")

            st.subheader("Dettaglio delle Probabilità:")
            # Ordina le probabilità dalla più alta alla più bassa per una migliore visualizzazione
            sorted_probs = sorted(all_probs.items(), key=lambda item: item[1], reverse=True)
            for category, prob in sorted_probs:
                st.write(f"- {category}: {prob:.2%}")
    else:
        st.warning("Per favore, inserisci del testo per la classificazione.") # Messaggio se la textbox è vuota

st.markdown("---")
st.info("Sviluppato con Streamlit e Hugging Face Transformers per PoisonChat.")
