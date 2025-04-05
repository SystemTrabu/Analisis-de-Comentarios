from models.extensions import db
from datetime import datetime


class Comentario(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    texto = db.Column(db.String(500), nullable=False) 
    autor = db.Column(db.String(100), nullable=False)  
    fecha_creacion = db.Column(db.String(100), nullable=False) 
    polaridad = db.Column(db.String(100), nullable=False)


    def __repr__(self):
        return f'<Comentario {self.id} - {self.autor}>'

