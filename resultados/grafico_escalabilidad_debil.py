import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo de resultados
df = pd.read_csv("resultados/numba_weak_runs.csv")

# Calcular el tiempo medio por configuración de threads
df_grouped = df.groupby('threads')['time_s'].mean().reset_index()

# Graficar
plt.figure(figsize=(8, 5))
plt.plot(df_grouped['threads'], df_grouped['time_s'], marker='o', linestyle='-', color='purple', label='Numba Paralelo (Escalabilidad Débil)')

# Agregar la línea teórica (tiempo constante = tiempo del caso P=1)
T1_weak = df_grouped.loc[df_grouped['threads'] == 1, 'time_s'].iloc[0]
plt.axhline(y=T1_weak, color='gray', linestyle='--', label=f'Tiempo Ideal (T1={T1_weak:.2f}s)')

plt.title('Escalabilidad Débil (Tiempo vs Threads)')
plt.xlabel('Threads (P)')
plt.ylabel('Tiempo medio (s)')
plt.xticks(df_grouped['threads'])
plt.grid(True)
plt.legend()
plt.show()