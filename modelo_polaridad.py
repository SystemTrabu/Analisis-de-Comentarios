import re
import unicodedata
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import BertForSequenceClassification, BertTokenizer, pipeline, MarianMTModel, MarianTokenizer
from langdetect import detect
import torch
from transformers import MarianMTModel, MarianTokenizer, pipeline
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load("es_core_news_sm")
spanish_stopwords = set(word.lower() for word in stopwords.words('spanish'))
spanish_stopwords.discard("si")
spanish_stopwords.discard("no")
spanish_stopwords.discard("sí")
spanish_stopwords.discard("todo")
spanish_stopwords.discard("es")
spanish_stopwords.discard("de")
spanish_stopwords.discard("esta")
spanish_stopwords.discard("este")
spanish_stopwords.discard("el")
spanish_stopwords.discard("la")
spanish_stopwords.discard("son")
spanish_stopwords.discard("sus")
spanish_stopwords.discard("para")



model_path = "./intelisis_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

def limpiar_texto(texto):
    texto = unicodedata.normalize('NFD', texto).encode('ascii', 'ignore').decode('utf-8')
    texto = re.sub(r'\d', 'd', texto)
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    texto = texto.lower().strip()
    
    tokens = word_tokenize(texto, language="spanish")
    texto_filtrado = [palabra for palabra in tokens if palabra not in spanish_stopwords]
    
    print(f"Texto limpio: {texto_filtrado}")
    return texto_filtrado
    


def traduccion(texto):
    idioma = detect(texto)
    
    if idioma == "es":
        return texto
        
    model_name = 'Helsinki-NLP/opus-mt-en-es'
    translation_model = MarianMTModel.from_pretrained(model_name)
    translation_tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    tokens = translation_tokenizer(texto, return_tensors="pt", padding=True)
    output = translation_model.generate(**tokens)
    texto_traducido = translation_tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(f"Texto traducido: {texto_traducido} ")
    return texto_traducido

def analizar_pol(texto):
    print(f"Texto en analizar pol ", texto)
    
    texto_unido = " ".join(texto)
    inputs = tokenizer(texto_unido, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class_idx = torch.argmax(logits, dim=1).item()
    
    sentiment_map = {0: "NEG", 1: "NEU", 2: "POS"}
    predicted_label = sentiment_map[predicted_class_idx]
    
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    confidence_score = probabilities[0][predicted_class_idx].item()
    
    return [{"label": predicted_label, "score": confidence_score}]

