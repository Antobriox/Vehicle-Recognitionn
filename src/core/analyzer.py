import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import os
from datetime import datetime
from collections import Counter
import numpy as np

class VehicleAnalyzer:
    def __init__(self):
        self.detections = []
        self.vehicle_counts = {}
        self.confidence_scores = []
        
    def add_detection(self, frame_number, class_id, class_name, confidence, track_id):
        self.detections.append({
            'frame': frame_number,
            'class_id': class_id,
            'class_name': class_name,
            'confidence': confidence,
            'track_id': track_id
        })
        
    def analyze_detections(self):
        df = pd.DataFrame(self.detections)
        
        # Crear directorio para guardar las gráficas
        os.makedirs('reports/images', exist_ok=True)
        
        # 1. Gráfica de conteo de vehículos por tipo
        plt.figure(figsize=(10, 6))
        vehicle_counts = df['class_name'].value_counts()
        sns.barplot(x=vehicle_counts.index, y=vehicle_counts.values)
        plt.title('Conteo de Vehículos por Tipo')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('reports/images/vehicle_counts.png')
        plt.close()
        
        # 2. Gráfica de distribución de confianza
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='confidence', bins=30)
        plt.title('Distribución de Scores de Confianza')
        plt.savefig('reports/images/confidence_distribution.png')
        plt.close()
        
        # 3. Gráfica de vehículos detectados a lo largo del tiempo
        plt.figure(figsize=(12, 6))
        df_timeline = df.groupby('frame')['class_name'].count()
        plt.plot(df_timeline.index, df_timeline.values)
        plt.title('Vehículos Detectados por Frame')
        plt.xlabel('Número de Frame')
        plt.ylabel('Cantidad de Vehículos')
        plt.savefig('reports/images/vehicles_timeline.png')
        plt.close()
        
        return {
            'total_vehicles': len(df['track_id'].unique()),
            'vehicle_counts': vehicle_counts.to_dict(),
            'avg_confidence': df['confidence'].mean(),
            'max_vehicles_frame': df_timeline.max()
        }
    
    def generate_pdf_report(self, stats):
        doc = SimpleDocTemplate(
            "reports/vehicle_analysis_report.pdf",
            pagesize=letter
        )
        
        styles = getSampleStyleSheet()
        story = []
        
        # Título
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        story.append(Paragraph("Reporte de Análisis de Vehículos", title_style))
        story.append(Spacer(1, 20))
        
        # Fecha del reporte
        story.append(Paragraph(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Estadísticas generales
        story.append(Paragraph("Estadísticas Generales:", styles['Heading2']))
        stats_data = [
            ["Total de vehículos detectados:", str(stats['total_vehicles'])],
            ["Confianza promedio:", f"{stats['avg_confidence']:.2%}"],
            ["Máximo de vehículos por frame:", str(stats['max_vehicles_frame'])]
        ]
        
        table = Table(stats_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 20))
        
        # Agregar gráficas
        for img_name in ['vehicle_counts.png', 'confidence_distribution.png', 'vehicles_timeline.png']:
            img_path = f'reports/images/{img_name}'
            if os.path.exists(img_path):
                img = Image(img_path, width=450, height=300)
                story.append(img)
                story.append(Spacer(1, 20))
        
        doc.build(story) 