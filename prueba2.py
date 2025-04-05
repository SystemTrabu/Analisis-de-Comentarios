import re
import unicodedata
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import MarianMTModel, MarianTokenizer, pipeline
import pandas as pd

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Cargar el modelo de spaCy una sola vez
nlp = spacy.load("es_core_news_sm")
spanish_stopwords = set(word.lower() for word in stopwords.words('spanish'))

class prueba2:
    @staticmethod
    def limpiar_texto(texto):
        # Normalizar el texto (eliminar acentos y caracteres especiales)
        texto = unicodedata.normalize('NFD', texto).encode('ascii', 'ignore').decode('utf-8')
        texto = re.sub(r'\d', 'd', texto)  # Reemplazar números por 'd'
        texto = re.sub(r'[^a-zA-Z\s]', '', texto)  # Eliminar caracteres no alfabéticos
        texto = texto.lower().strip()

        # Tokenizar el texto y eliminar stopwords
        tokens = word_tokenize(texto)
        texto_filtrado = [palabra for palabra in tokens if palabra not in spanish_stopwords]

        # Lematizar el texto
        doc = nlp(" ".join(texto_filtrado))
        lematizado = [token.lemma_ for token in doc]
        
        return lematizado

    @staticmethod
    def traduccion(texto):
        model_name = 'Helsinki-NLP/opus-mt-en-es'
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)

        tokens = tokenizer(texto, return_tensors="pt", padding=True)
        output = model.generate(**tokens)
        texto_traducido = tokenizer.decode(output[0], skip_special_tokens=True)

        return texto_traducido

# Función de conversión desde el archivo Excel
def conversion():
    # Cargar el archivo Excel
    df = pd.read_excel('frases_numeradas.xlsx')

    # Mapea las clases numéricas a sus etiquetas
    etiquetas = {1: '__label__usabilidad', 2: '__label__soporte', 3: '__label__rendimiento'}

    # Crear un archivo de texto con el formato adecuado para FastText
    with open('comentarios.txt', 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            etiqueta = etiquetas.get(row['Clase'])
            comentario = row['Comentario']
                
            # Llamar a la función limpiar_texto de la clase prueba2
            comentario = prueba2.limpiar_texto(comentario)  # Corregido para llamar correctamente al método estático
            
            if etiqueta:
                f.write(f"{etiqueta} {' '.join(comentario)}\n")  # Unir el texto lematizado y escribirlo en el archivo

if __name__ == "__main__":
    # Ejecutar la conversión para crear el archivo
    conversion()
    print("Archivo de texto creado con éxito.")
