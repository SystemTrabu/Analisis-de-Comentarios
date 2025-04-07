from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

def clasificar_categorias(texto):
    etiquetas = ["rendimiento", "soporte", "usabilidad", "sugerencias", "calidad", "odio", "preguntas"]
    
    resultado = classifier(texto, etiquetas, multi_label=True)
    
    primero = resultado['labels'][0]
    segundo = resultado['labels'][1] if resultado['scores'][1] >= 0.80 else None
    
    if segundo:
        return [primero, segundo]
    else:
        return [primero]
