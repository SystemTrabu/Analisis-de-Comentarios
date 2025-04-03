from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import unicodedata

# Función de limpieza de texto
def limpiar_texto(texto):
    if not isinstance(texto, str):
        return ""
    
    # Convertir a minúsculas
    texto = texto.lower()
    
    # Normalizar caracteres unicode (acentos, etc.)
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    
    # Eliminar URLs
    texto = re.sub(r'https?://\S+|www\.\S+', ' ', texto)
    
    # Normalizar menciones y hashtags
    texto = re.sub(r'@\w+', '@usuario', texto)
    texto = re.sub(r'#\w+', '#hashtag', texto)
    
    # Normalizar alargamientos de palabras (repetición de letras)
    texto = re.sub(r'([a-z])\1{2,}', r'\1\1', texto)  # Reduce repeticiones a máximo 2
    
    # Limpiar espacios en blanco extras
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    # Normalizar puntuación repetida
    texto = re.sub(r'([!?.,;:])\1+', r'\1', texto)
    
    return texto

# 1. Preparar el dataset
# Supongamos que tienes un CSV con columnas "texto" y "es_groseria" (0 o 1)
df = pd.read_csv("dataset/dataset.csv")

# Aplicar limpieza a los textos del dataset
df["texto_limpio"] = df["texto"].apply(limpiar_texto)

# Verificar que no haya textos vacíos después de la limpieza
df = df[df["texto_limpio"].str.len() > 0].reset_index(drop=True)

# Dividir en entrenamiento y validación
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# 2. Crear Dataset personalizado para PyTorch
class DatasetGroserias(Dataset):
    def __init__(self, textos, etiquetas, tokenizer, max_length=128):
        self.encodings = tokenizer(textos, truncation=True, padding=True, max_length=max_length)
        self.labels = etiquetas

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 3. Cargar BETO y configurarlo para clasificación
tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
model = AutoModelForSequenceClassification.from_pretrained(
    "dccuchile/bert-base-spanish-wwm-cased", 
    num_labels=2  
)

# 4. Crear datasets de entrenamiento y evaluación - Usamos texto_limpio en lugar de texto
train_dataset = DatasetGroserias(
    train_df["texto_limpio"].tolist(), 
    train_df["es_groseria"].tolist(), 
    tokenizer
)
eval_dataset = DatasetGroserias(
    eval_df["texto_limpio"].tolist(), 
    eval_df["es_groseria"].tolist(), 
    tokenizer
)

# 5. Configurar el entrenamiento
training_args = TrainingArguments(
    output_dir="./detector_groserias",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# 6. Definir métricas de evaluación
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = (preds == labels).mean()
    return {
        'accuracy': acc,
    }

# 7. Crear y ejecutar el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# 8. Entrenar el modelo
trainer.train()

# 9. Guardar el modelo entrenado
model.save_pretrained("./detector_groserias_final")
tokenizer.save_pretrained("./detector_groserias_final")

# 10. Función para usar el modelo entrenado - Ahora incluye limpieza de texto
def detectar_groserías(texto, modelo, tokenizador):
    # Limpiamos el texto de entrada usando la misma función
    texto_limpio = limpiar_texto(texto)
    
    # Tokenizamos y hacemos la predicción
    inputs = tokenizador(texto_limpio, return_tensors="pt", padding=True, truncation=True)
    outputs = modelo(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    
    return {
        "texto_original": texto,
        "texto_procesado": texto_limpio,
        "es_groseria": bool(probs.argmax().item()),
        "probabilidad": probs[0][1].item()
    }

# Ejemplo de uso
texto_prueba = "estoy cansada mentalmente, ya no se que hacer con mi vida"
resultado = detectar_groserías(texto_prueba, model, tokenizer)
print(resultado)