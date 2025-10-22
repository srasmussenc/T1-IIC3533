import pandas as pd
import sys

# Define el nombre del archivo de entrada y salida
INPUT_FILE = "resultados/numba_weak_runs.csv"
OUTPUT_FILE = "resultados/numba_weak_runs_promediado.csv"

try:
    # 1. Cargar el CSV que contiene las repeticiones
    df = pd.read_csv(INPUT_FILE)
    
except FileNotFoundError:
    print(f"Error: No se encontró el archivo de entrada '{INPUT_FILE}'.")
    sys.exit(1)

# 2. Agrupar por las columnas de configuración y calcular el promedio del tiempo
# Agrupamos por variant, threads, y primes_count, ya que son idénticos por configuración.
df_promediado = df.groupby(['variant', 'threads', 'primes_count'], as_index=False)['time_s'].mean()

# 3. Renombrar la columna de tiempo para reflejar el promedio
df_promediado = df_promediado.rename(columns={'time_s': 'time_s_avg'})

# 4. Guardar el nuevo CSV con los resultados promediados
df_promediado.to_csv(OUTPUT_FILE, index=False)

print("--- Análisis Completado ---")
print(f"Datos originales: {len(df)} filas")
print(f"Datos promediados: {len(df_promediado)} filas")
print(f"Resultados guardados en: {OUTPUT_FILE}")
print("\nPrimeras filas de los datos promediados:")
print(df_promediado.head())