
# Hackathon-Tec-2025

## **Descripción**
Este proyecto tiene como objetivo desarrollar un sistema de procesamiento y análisis de comentarios, este codigo fue utilizado en un Hackathon y fue ganador del primer lugar, cuenta varias funcionalidades como detección de sentimientos, categorización, traducción, filtrado de groserías, y generación de reportes. El sistema utiliza procesamiento de lenguaje natural (NLP) y aprendizaje automático para clasificar los comentarios de manera efectiva.

## **Requisitos**

- **Python**: Versión **menor a 3.13** (Evitar versiones 3.13 o superiores, ya que podrían generar problemas de compatibilidad).
- **Bibliotecas**: Las dependencias necesarias están especificadas en el archivo `requirements.txt`.

## **Instalación**


1. **Clonar el repositorio y configurar el entorno**:
   ```bash
   git clone [url]
   cd [dir]
   pip install -r requirements.txt
   python -m spacy download es_core_news_sm
   python entrenamiento_pos.py
   python app.py
   ```

---

## **Endpoints de la API**

### **1. `POST /procesar_comentarios/`**
   **Descripción**: Procesa los comentarios recibidos, realiza la categorización y análisis de sentimiento.

   **Estructura de entrada**:
   ```json
   {
     "comentarios": [
       {
         "comentario": "Entrada STR",
         "fecha": "YYYY-MM-DD",
         "usuario": "USER",
         "hora": "HORA"
       }
     ]
   }
   ```

   **Estructura de salida**:
   ```json
   {
     "procesados": [
       {
         "categorias": ["CATEGORIA"],
         "comentario": "COMENTARIO STR",
         "procesado": "COMENTARIO PROCESADO STR",
         "sentimiento": "SENTIMIENTO",
         "usuario": "USER"
       }
     ]
   }
   ```

---

### **2. `GET /getCommentsPrin/`**
   **Descripción**: Devuelve todos los comentarios y estadísticas de los mismos.

   **Salida**:
   ```json
   {
     "comentarios": [
       {
         "categoria": [
           { "categoria_id": ID, "categoria_nombre": "CATEGORIA" },
           { "categoria_id": ID, "categoria_nombre": "CATEGORIA" }
         ]
       }
     ],
     "palabras_mas_usadas": [
       ["palabra", 10]
     ],
     "total_comentarios": TOTAL,
     "total_comentarios_neg": TOTAL,
     "total_comentarios_neu": TOTAL,
     "total_comentarios_pos": TOTAL
   }
   ```

---

### **3. `POST /grafica_sentimiento/`**
   **Descripción**: Genera datos para crear una gráfica de tendencias de sentimientos.

   **Estructura de entrada**:
   ```json
   {
     "fecha_inicio": "YYYY-MM-DD",
     "fecha_fin": "YYYY-MM-DD"
   }
   ```

   **Estructura de salida**:
   ```json
   {
     "dias": ["YYYY-MM-DD", "YYYY-MM-DD"],
     "sentimientos_promedio": [0.2, 0.6],
     "tendencia": [1, 2]
   }
   ```

---

### **4. `GET /reporteAll/`**
   **Descripción**: Genera un reporte en formato PDF de todos los comentarios procesados.

---

### **5. `GET /getComentsPos/`**
   **Descripción**: Devuelve todos los comentarios positivos y neutrales.

   **Salida**:
   ```json
   [
     {
       "comentario": "COMENTARIO STR",
       "fecha": "YYYY-MM-DD",
       "id": ID_COMENTARIO
     }
   ]
   ```

---

## **Funcionalidades**

- **Detección de sentimientos**: Clasificación de comentarios como positivos, negativos o neutrales.
- **Categorización**: Asigna categorías a los comentarios procesados.
- **Traducción inglés - español**: Traducción automática de comentarios entre inglés y español.
- **Detección y censura de groserías**: Identificación y censura de palabras inapropiadas.
- **Filtrado de groserías**: El sistema filtra comentarios que contienen lenguaje ofensivo.
- **Generación de reportes**: Crea reportes en formato PDF con los comentarios procesados y sus análisis.

---




