import pandas as pd
import prueba2  # Asegúrate de que esta importación esté bien


f = pd.read_excel('frases_numeradas.xlsx')
# Mapea las clases numéricas a sus etiquetas
etiquetas = {1: '__label__usabilidad', 2: '__label__soporte', 3: '__label__rendimiento',  4: '__label__neutro'}

        # Crear un archivo de texto con el formato adecuado para FastText
# Cambia la apertura del archivo para usar UTF-8
with open('comentarios.txt', 'w', encoding='utf-8') as f:
    for _, row in f.iterrows():
        etiqueta = etiquetas.get(row['Clase'])
        comentario = row['Comentario']
        if etiqueta:
            f.write(f"{etiqueta} {comentario}\n")
