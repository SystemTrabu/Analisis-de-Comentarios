from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
import torch

# Cargar el dataset desde el archivo JSON con etiquetas numéricas
dataset = load_dataset("json", data_files="dataset/intelisis_comments.json", field="data")

# Crear una partición para validación (si no tienes una ya)
dataset = dataset["train"].train_test_split(test_size=0.1)  # 10% para validación

# Mostrar una muestra de los datos para verificar
print("Ejemplo de dato:")
print(dataset["train"][0])

# Cargar el modelo y el tokenizador preentrenado
model_name = "finiteautomata/beto-sentiment-analysis"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 clases: NEG(0), NEU(1), POS(2)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Tokenizar el dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Renombrar la columna 'label' a 'labels' (requerido por Transformers)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# Verificar que las columnas correctas estén presentes
print("\nColumnas en el dataset tokenizado:")
print(tokenized_datasets["train"].column_names)
print("\nEjemplo de dato tokenizado:")
print(tokenized_datasets["train"][0])

# Configurar los parámetros de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",          # Usa eval_strategy en lugar de evaluation_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Asegurarnos de que solo se pasen las columnas necesarias
if 'text' in tokenized_datasets['train'].column_names:
    tokenized_datasets = tokenized_datasets.remove_columns(['text'])  # Eliminar la columna 'text' si existe

# Usar Trainer para entrenar el modelo
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Entrenar el modelo
trainer.train()

# Guardar el modelo entrenado
model.save_pretrained("./intelisis_model")
tokenizer.save_pretrained("./intelisis_model")

# Añadir código para usar el modelo entrenado para predecir sentimientos
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    sentiment_map = {0: "Negativo", 1: "Neutro", 2: "Positivo"}
    return sentiment_map[predicted_class]

# Ejemplos de uso
examples = [
    "Esta cabronsisimo su producto",
    "Esta culero su producto, no vuelvo a comprar",
    "Vayanse a la mierda",
    "Ustedes son demasiado buenos!!"
]

print("\nPrueba del modelo con ejemplos:")
for example in examples:
    sentiment = predict_sentiment(example)
    print(f"Texto: '{example}' - Sentimiento: {sentiment}")