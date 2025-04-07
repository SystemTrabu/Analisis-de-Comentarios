import re
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
import torch

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load stopwords and process them
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

# Preprocessing function that matches your application logic
def preprocess_text(texto):
    texto = unicodedata.normalize('NFD', texto).encode('ascii', 'ignore').decode('utf-8')
    texto = re.sub(r'\d', 'd', texto)
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    texto = texto.lower().strip()
    
    tokens = word_tokenize(texto, language="spanish")
    texto_filtrado = [palabra for palabra in tokens if palabra not in spanish_stopwords]
    
    return " ".join(texto_filtrado)

# Load the dataset from the JSON file with numeric labels
dataset = load_dataset("json", data_files="dataset/intelisis_comments.json", field="data")

# Create a validation split if you don't already have one
dataset = dataset["train"].train_test_split(test_size=0.1)  # 10% for validation

# Show a sample of the data to verify
print("Example data:")
print(dataset["train"][0])

# Load the pretrained model and tokenizer
model_name = "finiteautomata/beto-sentiment-analysis"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 classes: NEG(0), NEU(1), POS(2)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Define the tokenization function with preprocessing
def tokenize_function(examples):
    # First apply the same preprocessing as in your application
    preprocessed_texts = [preprocess_text(text) for text in examples["text"]]
    # Then tokenize with BERT tokenizer
    return tokenizer(preprocessed_texts, padding="max_length", truncation=True, max_length=128)

# Apply tokenization to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Rename the 'label' column to 'labels' (required by Transformers)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# Verify that the correct columns are present
print("\nColumns in the tokenized dataset:")
print(tokenized_datasets["train"].column_names)
print("\nExample of tokenized data:")
print(tokenized_datasets["train"][0])

# Configure the training parameters
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",          # Use eval_strategy instead of evaluation_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Make sure only the necessary columns are passed
if 'text' in tokenized_datasets['train'].column_names:
    tokenized_datasets = tokenized_datasets.remove_columns(['text'])  # Remove the 'text' column if it exists

# Use Trainer to train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained("./intelisis_model")
tokenizer.save_pretrained("./intelisis_model")

def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    
    inputs = tokenizer(preprocessed_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    sentiment_map = {0: "Negativo", 1: "Neutro", 2: "Positivo"}
    return sentiment_map[predicted_class]

examples = [
    "Esta cabronsisimo su producto",
    "Esta culero su producto, no vuelvo a comprar",
    "Vayanse a la mierda",
    "Ustedes son demasiado buenos!!"
]

print("\nModel test with examples:")
for example in examples:
    sentiment = predict_sentiment(example)
    print(f"Text: '{example}' - Sentiment: {sentiment}")