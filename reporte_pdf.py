from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import pandas as pd
from reportlab.pdfgen import canvas

def generar_reporte(comentarios, comentarios_pos, comentarios_neu, comentarios_neg):
    comentarios_data = {
        'id': [],
        'comentario': [],
        'categoria': [],
        'polaridad': [],
    }

    # Procesar los comentarios positivos
    for comentario_pos in comentarios_pos:
        comentario = comentario_pos.comentario
        categorias = [cat.categoria for cat in comentario.categoria_comen]  
        comentarios_data['id'].append(comentario.id)
        comentarios_data['comentario'].append(comentario.comentario)
        comentarios_data['categoria'].append(categorias)
        comentarios_data['polaridad'].append('positivo')

    # Procesar los comentarios negativos
    for comentario_neg in comentarios_neg:
        comentario = comentario_neg.comentario
        categorias = [cat.categoria for cat in comentario.categoria_comen]
        comentarios_data['id'].append(comentario.id)
        comentarios_data['comentario'].append(comentario.comentario)
        comentarios_data['categoria'].append(categorias)
        comentarios_data['polaridad'].append('negativo')

    # Procesar los comentarios neutrales
    for comentario_neu in comentarios_neu:
        comentario = comentario_neu.comentario
        categorias = [cat.categoria for cat in comentario.categoria_comen]
        comentarios_data['id'].append(comentario.id)
        comentarios_data['comentario'].append(comentario.comentario)
        comentarios_data['categoria'].append(categorias)
        comentarios_data['polaridad'].append('neutral')

    # Crear un DataFrame de Pandas
    df = pd.DataFrame(comentarios_data)

    # Contar comentarios por polaridad
    total_pos = len(comentarios_pos)
    total_neg = len(comentarios_neg)
    total_neu = len(comentarios_neu)
    total_comentarios = total_pos + total_neg + total_neu

    # Contar los comentarios por categoría
    categorias_dict = {}
    for index, row in df.iterrows():
        categorias_lista = row['categoria']
        for cat in categorias_lista:
            
            cat_nombre = cat
            if hasattr(cat, 'categoria'):  
                cat_nombre = cat.categoria
            
            cat_nombre = str(cat_nombre)
            
            if cat_nombre in categorias_dict:
                categorias_dict[cat_nombre] += 1
            else:
                categorias_dict[cat_nombre] = 1

    reporte_pdf = 'reporte_comentarios.pdf'
    doc = SimpleDocTemplate(reporte_pdf, pagesize=letter)
    
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    subtitle_style = styles['Heading2']
    normal_style = styles['Normal']
    
    comentario_style = ParagraphStyle(
        'ComentarioStyle',
        parent=normal_style,
        fontSize=9,
        leading=10,  
        spaceAfter=6,
        wordWrap='CJK'  
    )
    
    elements.append(Paragraph("Reporte de Comentarios", title_style))
    elements.append(Spacer(1, 12))
    
    elements.append(Paragraph("Resumen de Datos", subtitle_style))
    elements.append(Spacer(1, 6))
    
    resumen_data = [
        ["Tipo de comentario", "Cantidad", "Porcentaje"],
        ["Positivos", str(total_pos), f"{(total_pos/total_comentarios*100):.1f}%" if total_comentarios > 0 else "0%"],
        ["Negativos", str(total_neg), f"{(total_neg/total_comentarios*100):.1f}%" if total_comentarios > 0 else "0%"],
        ["Neutrales", str(total_neu), f"{(total_neu/total_comentarios*100):.1f}%" if total_comentarios > 0 else "0%"],
        ["Total", str(total_comentarios), "100%"]
    ]
    
    resumen_table = Table(resumen_data, colWidths=[150, 100, 100])
    resumen_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(resumen_table)
    elements.append(Spacer(1, 12))
    
    elements.append(Paragraph("Categorías más frecuentes", subtitle_style))
    elements.append(Spacer(1, 6))
    
    categorias_ordenadas = sorted(categorias_dict.items(), key=lambda x: x[1], reverse=True)
    
    cat_data = [["Categoría", "Cantidad"]]
    for cat, count in categorias_ordenadas[:10]:  
        cat_data.append([cat, str(count)])
    
    cat_table = Table(cat_data, colWidths=[200, 100])
    cat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(cat_table)
    elements.append(Spacer(1, 20))
    
    elements.append(Paragraph("Comentarios Positivos", subtitle_style))
    elements.append(Spacer(1, 6))

    comentarios_pos_data = [['ID', 'Comentario', 'Categorías']]
    for comentario_pos in comentarios_pos:
        comentario = comentario_pos.comentario
        comentario_text = comentario.comentario

        categorias = []
        for cat in comentario.categoria_comen:
            if hasattr(cat, 'categoria'):
                categorias.append(str(cat.categoria.categoria))
            else:
                categorias.append(str(cat))
        
        categoria_text = ', '.join(categorias) if categorias else 'Sin categoría'

        if len(comentario_text) > 300:
            comentario_text = comentario_text[:297] + '...'
        comentario_paragraph = Paragraph(comentario_text, comentario_style)

        comentarios_pos_data.append([comentario.id, comentario_paragraph, categoria_text])
    
    if len(comentarios_pos_data) > 1: 
        pos_table = Table(
            comentarios_pos_data, 
            colWidths=[40, 300, 180],
            repeatRows=1
        )
        
        pos_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.green),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (2, 0), (2, -1), 'CENTER'),
            ('ALIGN', (1, 0), (1, 0), 'CENTER'),
            ('ALIGN', (1, 1), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ('ROWHEIGHT', (0, 1), (-1, -1), None),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
        ]))
        
        elements.append(pos_table)
    else:
        elements.append(Paragraph("No hay comentarios positivos para mostrar.", normal_style))
    
    elements.append(Spacer(1, 20))
    
    elements.append(Paragraph("Comentarios Neutrales", subtitle_style))
    elements.append(Spacer(1, 6))

    comentarios_neu_data = [['ID', 'Comentario', 'Categorías']]
    for comentario_neu in comentarios_neu:
        comentario = comentario_neu.comentario
        comentario_text = comentario.comentario

        categorias = []
        for cat in comentario.categoria_comen:
            if hasattr(cat, 'categoria'):
                categorias.append(str(cat.categoria.categoria))
            else:
                categorias.append(str(cat))
        
        categoria_text = ', '.join(categorias) if categorias else 'Sin categoría'

        if len(comentario_text) > 300:
            comentario_text = comentario_text[:297] + '...'
        comentario_paragraph = Paragraph(comentario_text, comentario_style)

        comentarios_neu_data.append([comentario.id, comentario_paragraph, categoria_text])
    
    if len(comentarios_neu_data) > 1:  
        neu_table = Table(
            comentarios_neu_data, 
            colWidths=[40, 300, 180],
            repeatRows=1
        )
        
        neu_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (2, 0), (2, -1), 'CENTER'),
            ('ALIGN', (1, 0), (1, 0), 'CENTER'),
            ('ALIGN', (1, 1), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ('ROWHEIGHT', (0, 1), (-1, -1), None),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
        ]))
        
        elements.append(neu_table)
    else:
        elements.append(Paragraph("No hay comentarios neutrales para mostrar.", normal_style))
    
    elements.append(Spacer(1, 20))
    
    elements.append(Paragraph("Comentarios Negativos", subtitle_style))
    elements.append(Spacer(1, 6))

    comentarios_neg_data = [['ID', 'Comentario', 'Categorías', 'Palabra no censurada']]
    for comentario_neg in comentarios_neg:
        comentario = comentario_neg.comentario
        comentario_text = comentario.comentario

        categorias = []
        for cat in comentario.categoria_comen:
            if hasattr(cat, 'categoria'):
                categorias.append(str(cat.categoria.categoria))
            else:
                categorias.append(str(cat))
        
        categoria_text = ', '.join(categorias) if categorias else 'Sin categoría'

        if len(comentario_text) > 300:
            comentario_text = comentario_text[:297] + '...'
        comentario_paragraph = Paragraph(comentario_text, comentario_style)

        palabra_no_censurada = comentario_neg.comentario_sincensura if hasattr(comentario_neg, 'comentario_sincensura') else 'N/A'
        palabra_paragraph = Paragraph(str(palabra_no_censurada), comentario_style)


        comentarios_neg_data.append([comentario.id, comentario_paragraph, categoria_text, palabra_paragraph])
        
    if len(comentarios_neg_data) > 1:  
        neg_table = Table(
            comentarios_neg_data, 
            colWidths=[30, 240, 140, 140],
            repeatRows=2
        )
        
        neg_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.red),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (2, 0), (3, -1), 'CENTER'),
            ('ALIGN', (1, 0), (1, 0), 'CENTER'),
            ('ALIGN', (1, 1), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ('ROWHEIGHT', (0, 1), (-1, -1), None),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
        ]))
        
        elements.append(neg_table)
    else:
        elements.append(Paragraph("No hay comentarios negativos para mostrar.", normal_style))
    
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Detalle de Todos los Comentarios", subtitle_style))
    elements.append(Spacer(1, 6))
    
    comentarios_data = [['ID', 'Comentario', 'Categorías', 'Polaridad']]
    
    for index, row in df.iterrows():
        comentario_text = row['comentario']
        if len(comentario_text) > 300:
            comentario_text = comentario_text[:297] + '...'
        
        comentario_paragraph = Paragraph(comentario_text, comentario_style)
        
        categorias_lista = row['categoria']
        categoria_textos = []
        
        for cat in categorias_lista:
            if hasattr(cat, 'categoria'):
                categoria_textos.append(str(cat.categoria))
            else:
                categoria_textos.append(str(cat))
                
        categoria_text = ', '.join(categoria_textos) if categoria_textos else 'Sin categoría'
        
        polaridad_text = row['polaridad']
        comentarios_data.append([row['id'], comentario_paragraph, categoria_text, polaridad_text])
    
    comentarios_table = Table(
        comentarios_data, 
        colWidths=[40, 250, 150, 80],
        repeatRows=1  
    )
    
    comentarios_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'), 
        ('ALIGN', (2, 0), (3, -1), 'CENTER'),  
        ('ALIGN', (1, 0), (1, 0), 'CENTER'),   
        ('ALIGN', (1, 1), (1, -1), 'LEFT'),    
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('ROWHEIGHT', (0, 1), (-1, -1), None),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
    ]))
    
    for i in range(1, len(comentarios_data)):
        polaridad = comentarios_data[i][3]
        if polaridad == 'positivo':
            comentarios_table.setStyle(TableStyle([('BACKGROUND', (3, i), (3, i), colors.lightgreen)]))
        elif polaridad == 'negativo':
            comentarios_table.setStyle(TableStyle([('BACKGROUND', (3, i), (3, i), colors.lightcoral)]))
        elif polaridad == 'neutral':
            comentarios_table.setStyle(TableStyle([('BACKGROUND', (3, i), (3, i), colors.lightgrey)]))
    
    elements.append(comentarios_table)
    
    doc.build(elements)
    
    return reporte_pdf