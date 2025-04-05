from datetime import datetime
import re
from flask import Flask, request, jsonify
from matplotlib import pyplot as plt
from models.comentarios import Categoria, CategoriaComen, db, Comentario
from models.comentarios import db, Usuario, Comentario, ComentarioPos, ComentarioNeg, ComentarioNeu
from prueba2 import traduccion, limpiar_texto, analizar_pol
from groserias import censurar_groseria
from collections import Counter
from categorias import clasificar_categorias
from flask_migrate import Migrate
from collections import defaultdict



app = Flask(__name__)

# Configuración de MySQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:1234@localhost/comentarios'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Inicializar db con app
db.init_app(app)
migrate = Migrate(app, db)
@app.route('/')
def home():
    return "¡API Flask funcionando!"

def palabrasUsadas(comentarios):
    texto_unido = " ".join([comentario.comentario.comentario for comentario in comentarios]) 

    # Extraer las palabras
    palabras = re.findall(r'\b\w+\b', texto_unido.lower())

    # Lista de palabras a excluir (stopwords)
    stopwords = {"el", "la", "es", "muy", "pero", "no", "nada", "si", "ellos", "su", "son", "lo","me","sus","y"}
    
    # Filtrar las palabras eliminando las stopwords
    palabras_filtradas = [p for p in palabras if p not in stopwords]

    # Contar la frecuencia de las palabras
    contador = Counter(palabras_filtradas)

    # Obtener las 10 palabras más comunes
    palabras_mas_comunes = contador.most_common(10)
    
    return palabras_mas_comunes

@app.route('/getCommentsPrin/')
def get_coments():
    comentarios = Comentario.query.all()
    comentarios_pos = ComentarioPos.query.all()
    comentarios_neg = ComentarioNeg.query.all()
    comentarios_neu = ComentarioNeu.query.all()

    palabras_mas_usadas = palabrasUsadas(comentarios_pos)

    total_comentarios = len(comentarios)
    total_comentarios_pos = len(comentarios_pos)
    total_comentarios_neg = len(comentarios_neg)
    total_comentarios_neu = len(comentarios_neu)

    respuesta = {
        "total_comentarios": total_comentarios,
        "total_comentarios_pos": total_comentarios_pos,
        "total_comentarios_neg": total_comentarios_neg,
        "total_comentarios_neu": total_comentarios_neu,
        "palabras_mas_usadas": palabras_mas_usadas,
        "comentarios": [
            {
                "id": comentario.id,
                "comentario": comentario.comentario,
                "categoria": "positivo" if comentario.id in [c.id_comentario for c in comentarios_pos] else ("negativo" if comentario.id in [c.id_comentario for c in comentarios_neg] else "neutral")
            }
            for comentario in comentarios
        ]
    }

    return jsonify(respuesta)


@app.route('/procesar_comentarios/', methods=['POST'])
def procesar_comentarios():
    data = request.get_json()

    if not data or "comentarios" not in data:
        return jsonify({"error": "Faltan los comentarios en el cuerpo del request"}), 400

    comentarios = data["comentarios"]
    resultados = []

    for item in comentarios:
        texto = item["comentario"]
        fecha = item["fecha"]
        hora = item["hora"]
        
        nombre_usuario = item.get("usuario", "").strip() or "Anonimo"

        usuario = Usuario.query.filter_by(usuario=nombre_usuario).first()
        if not usuario:
            usuario = Usuario(usuario=nombre_usuario)
            db.session.add(usuario)
            db.session.commit()  

        texto_traducido = traduccion(texto)
        texto_limpio = limpiar_texto(texto_traducido)
        resultado = analizar_pol(texto_limpio)
        sentimiento = resultado[0]['label']
        categorias = clasificar_categorias(texto_traducido)
        print(f"Las categorias del texto son: {categorias}")
        comentario_sincensura = texto_traducido  
        
        if sentimiento == 'NEG':
            for palabra in texto_limpio:
                if len(palabra) != 1:
                    texto_traducido = texto_traducido.replace(palabra, censurar_groseria(palabra))

        # Crear el objeto comentario
        comentario_obj = Comentario(
            comentario=texto_traducido,
            id_usuario=usuario.id,
            fecha=datetime.strptime(fecha, "%Y-%m-%d").date(),
            hora=datetime.strptime(hora, "%H:%M").time()
        )
        db.session.add(comentario_obj)
        db.session.commit()  # Necesario para obtener el ID del comentario

        # Insertar en la tabla de sentimiento correspondiente
        if sentimiento == "POS":
            db.session.add(ComentarioPos(id_comentario=comentario_obj.id))
        elif sentimiento == "NEG":
            db.session.add(ComentarioNeg(id_comentario=comentario_obj.id, comentario_sincensura=comentario_sincensura))
        else:
            db.session.add(ComentarioNeu(id_comentario=comentario_obj.id))

        # Asociar las categorías al comentario usando la tabla intermedia 'categoria_comen'
        for categoria_nombre in categorias:
            print(f"Revisando categoria: {categoria_nombre}")
            categoria = Categoria.query.filter_by(categoria=categoria_nombre).first()
            print(f"Encontró : {categoria}")
            if not categoria:
                categoria = Categoria(categoria=categoria_nombre)
                db.session.add(categoria)
                db.session.commit()
                print("No la encontró, la ha creado")

            # Crear la relación en la tabla intermedia categoria_comen
            categoria_comen_obj = CategoriaComen(id_comentario=comentario_obj.id, id_categoria=categoria.id)
            db.session.add(categoria_comen_obj)
        
        db.session.commit()  # Realizar el commit para guardar todas las relaciones

        resultados.append({
            "usuario": nombre_usuario,
            "comentario": texto,
            "procesado": texto_traducido,
            "sentimiento": sentimiento,
            "categorias": categorias
        })

    return jsonify({"procesados": resultados}), 200


@app.route('/grafica_sentimiento/', methods=['POST'])
def grafica_sentimiento():
    # Recibir las fechas de inicio y fin desde el frontend
    data = request.get_json()
    fecha_inicio = data.get('fecha_inicio')
    fecha_fin = data.get('fecha_fin')

    # Validar que ambas fechas hayan sido enviadas
    if not fecha_inicio or not fecha_fin:
        return jsonify({"error": "Fechas no proporcionadas"}), 400
    
    try:
        # Convertir las fechas de string a objetos datetime
        fecha_inicio_obj = datetime.strptime(fecha_inicio, '%Y-%m-%d').date()
        fecha_fin_obj = datetime.strptime(fecha_fin, '%Y-%m-%d').date()

        # Filtrar los comentarios por el rango de fechas
        comentarios = Comentario.query.filter(Comentario.fecha >= fecha_inicio_obj, Comentario.fecha <= fecha_fin_obj).all()

        if not comentarios:
            return jsonify({"error": "No hay comentarios para el rango de fechas seleccionado"}), 404

        # Inicializar un diccionario para agrupar los comentarios por fecha
        comentarios_por_fecha = defaultdict(list)

        # Agrupar los comentarios por fecha
        for comentario in comentarios:
            # Calcular el sentimiento para el comentario
            if comentario.pos:
                sentimiento = 1  # Positivo
            elif comentario.neg:
                sentimiento = -1  # Negativo
            elif comentario.neu:
                sentimiento = 0  # Neutro
            else:
                continue  # Si no tiene un sentimiento asignado, lo omitimos

            comentarios_por_fecha[comentario.fecha].append(sentimiento)

        # Inicializar listas para los días y los sentimientos promedio
        dias = []
        sentimientos_promedio = []

        # Calcular el sentimiento promedio para cada día
        for fecha, sentimientos in comentarios_por_fecha.items():
            dias.append(str(fecha))  # Convertir la fecha a string para la gráfica
            sentimiento_promedio = sum(sentimientos) / len(sentimientos)  # Promedio del sentimiento
            sentimientos_promedio.append(sentimiento_promedio)

        # Calcular la tendencia: diferencia de sentimiento entre días consecutivos
        tendencia = []
        for i in range(1, len(sentimientos_promedio)):
            # La tendencia entre dos días consecutivos es la diferencia de sentimientos
            diferencia = sentimientos_promedio[i] - sentimientos_promedio[i - 1]
            tendencia.append(diferencia)

        # Devolver los datos en formato JSON
        return jsonify({
            'dias': dias,
            'sentimientos_promedio': sentimientos_promedio,
            'tendencia': tendencia
        })

    except ValueError:
        return jsonify({"error": "Formato de fecha inválido"}), 400



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
 

    app.run(debug=True)
