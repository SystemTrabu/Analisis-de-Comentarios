import re
import unicodedata
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import MarianMTModel, MarianTokenizer, pipeline

# Descargar recursos de NLTK
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('punkt')

# Cargar stopwords una vez y procesarlas
spanish_stopwords = set(word.lower() for word in stopwords.words('spanish'))

# Cargar spaCy una vez
nlp = spacy.load("es_core_news_sm")

def limpiar_texto(texto):
    texto = unicodedata.normalize('NFD', texto).encode('ascii', 'ignore').decode('utf-8')
    texto = re.sub(r'\d', 'd', texto)
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    texto = texto.lower().strip()

    tokens = word_tokenize(texto)
    texto_filtrado = [palabra for palabra in tokens if palabra not in spanish_stopwords]

    doc = nlp(" ".join(texto_filtrado))
    lematizado = [token.lemma_ for token in doc]
    
    return lematizado

def traduccion(texto):
    model_name = 'Helsinki-NLP/opus-mt-en-es'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    tokens = tokenizer(texto, return_tensors="pt", padding=True)
    output = model.generate(**tokens)
    texto_traducido = tokenizer.decode(output[0], skip_special_tokens=True)

    return texto_traducido

# Texto de prueba
texto_prueba = "I'm mentally tired, I don't know what to do with my life anymore."

# Traducción y limpieza
TextoTraducido = traduccion(texto_prueba)
texto = limpiar_texto(TextoTraducido)

print(texto)
# modelo de analisis
analisis = pipeline("sentiment-analysis", model="finiteautomata/beto-sentiment-analysis")

texto = " ".join(texto)
resultado = analisis(texto)
print(texto)
print(resultado)
print(TextoTraducido)
