import fasttext

# Entrenar el modelo de clasificación
model = fasttext.train_supervised(input="comentarios.txt")

# Evaluar el modelo (opcional)
print("Precisión en el test:", model.test("comentarios_test.txt"))

# Realizar una predicción
comentario_nuevo = "La interfaz de usuario es muy intuitiva."
categoria_predicha = model.predict(comentario_nuevo)
print(f"El comentario pertenece a la categoría: {categoria_predicha[0][0]}")
