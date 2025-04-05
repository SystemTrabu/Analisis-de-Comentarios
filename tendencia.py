import matplotlib.pyplot as plt
import numpy as np

sentimientos = [1, -1, 0, 1, 0, -1, -1, 1, 0, 1, 0, -1]

# Supongamos que estos son los días o el tiempo (puede ser cualquier unidad temporal que elijas)
dias = np.arange(1, len(sentimientos) + 1)

# Crear la gráfica
plt.figure(figsize=(10, 6))
plt.plot(dias, sentimientos, marker='o', linestyle='-', color='b', label='Sentimiento')

# Personalización de la gráfica
plt.title('Tendencia de Sentimiento a lo Largo del Tiempo')
plt.xlabel('Día')
plt.ylabel('Sentimiento')
plt.xticks(dias)  # Marcar todos los días
plt.yticks([-1, 0, 1])  # Solo mostrar -1, 0, 1 en el eje y
plt.axhline(0, color='black',linewidth=1)  # Línea en el 0 para separar negativos de positivos
plt.grid(True)
plt.legend()

# Mostrar la gráfica
plt.show()
