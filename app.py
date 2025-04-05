from flask import Flask, request, jsonify
from detectar_groserias import detectar_groserías
from prueba2 import traduccion, limpiar_texto, analizar_pol
from groserias import censurar_groseria
#from flask_sqlalchemy import SQLAlchemy
#from models.extensions import db  
#from models.comentarios import Comentario  


app = Flask(__name__)

#app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:1234@localhost/intelisis'
#app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False 
#db.init_app(app)


@app.route('/')
def home():
    return "¡API Flask funcionando!"


@app.route('/procesar_comentarios/', methods=['POST'])
def procesar_comentarios():
    data = request.get_json()
    
    if not data or "comentarios" not in data:
        return jsonify({"error": "Faltan los comentarios en el cuerpo del request"}), 400

    comentarios = data["comentarios"]

    textos = [comentario["comentario"] for comentario in comentarios]
    for texto in textos:
        texto_traducido=traduccion(texto)
        texto_limpio=limpiar_texto(texto_traducido)
        resultado=analizar_pol(texto_limpio)

        print(f"Texto normal: {texto_traducido} ")
        print(resultado)
        
        if resultado[0]['label'] == 'NEG':
            
            for palabra in texto_limpio:
                if len(palabra)!=1:
                    texto_traducido.replace(palabra, censurar_groseria(palabra))
        

        print(f"Texto limpio despues de censurar {texto_traducido}")
        

    
    
        
        #nuevo_comentario = Comentario(texto=texto, autor="prueba", fecha_creacion="fechaprueba", polaridad=resultado)
        #b.session.add(nuevo_comentario)
        #db.session.commit()



    return "Okey"

#with app.app_context():
 #   db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
