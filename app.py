from flask import Flask, request, jsonify
from detectar_groserias import detectar_groserías
app = Flask(__name__)

@app.route('/')
def home():
    return "¡API Flask funcionando!"


@app.route('/procesar_comentarios/')
def procesar_comentarios(request):
    comentarios=request["body"]["comentarios"]
    
    for comentario in comentarios:
        detectar_groserías(comentario)
    


if __name__ == '__main__':
    app.run(debug=True)
