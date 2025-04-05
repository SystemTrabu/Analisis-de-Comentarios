from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_migrate import Migrate

db = SQLAlchemy()

class Usuario(db.Model):
    __tablename__ = 'usuarios'

    id = db.Column(db.Integer, primary_key=True)
    usuario = db.Column(db.String(100), nullable=False)

    comentarios = db.relationship('Comentario', backref='usuario', lazy=True)


class CategoriaComen(db.Model):
    __tablename__ = 'categoria_comen'

    id = db.Column(db.Integer, primary_key=True)
    id_comentario = db.Column(db.Integer, db.ForeignKey('comentarios.id'), nullable=False)
    id_categoria = db.Column(db.Integer, db.ForeignKey('categorias.id'), nullable=False)

    # Relaciones inversas
    comentario = db.relationship('Comentario', backref=db.backref('categoria_comen', lazy=True))
    categoria = db.relationship('Categoria', backref=db.backref('categoria_comen', lazy=True))


class Categoria(db.Model):
    __tablename__ = 'categorias'
    
    id = db.Column(db.Integer, primary_key=True)
    categoria = db.Column(db.String(100), nullable=False, unique=True)

    # Se eliminó el atributo comentarios aquí
    # comentarios = db.relationship('Comentario', backref='categoria', lazy=True)

class Comentario(db.Model):
    __tablename__ = 'comentarios'

    id = db.Column(db.Integer, primary_key=True)
    comentario = db.Column(db.Text, nullable=False)
    id_usuario = db.Column(db.Integer, db.ForeignKey('usuarios.id'), nullable=False)
    fecha = db.Column(db.Date, nullable=False, default=datetime.utcnow)
    hora = db.Column(db.Time, nullable=False, default=datetime.utcnow().time)
    
    # Relaciones con tablas de clasificación
    pos = db.relationship('ComentarioPos', backref='comentario', lazy=True)
    neg = db.relationship('ComentarioNeg', backref='comentario', lazy=True)
    neu = db.relationship('ComentarioNeu', backref='comentario', lazy=True)




class ComentarioPos(db.Model):
    __tablename__ = 'comentarios_pos'

    id = db.Column(db.Integer, primary_key=True)
    id_comentario = db.Column(db.Integer, db.ForeignKey('comentarios.id'), nullable=False)


class ComentarioNeg(db.Model):
    __tablename__ = 'comentarios_neg'

    id = db.Column(db.Integer, primary_key=True)
    id_comentario = db.Column(db.Integer, db.ForeignKey('comentarios.id'), nullable=False)
    comentario_sincensura = db.Column(db.Text, nullable=False)


class ComentarioNeu(db.Model):
    __tablename__ = 'comentarios_neu'

    id = db.Column(db.Integer, primary_key=True)
    id_comentario = db.Column(db.Integer, db.ForeignKey('comentarios.id'), nullable=False)



