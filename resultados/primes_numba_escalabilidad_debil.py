# primes_numba.py
import time
import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads

# ---- implementacion serial en python (para validar) ----
def is_prime_py(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

def count_primes_py(limit):
    c = 0
    for n in range(2, limit):
        if is_prime_py(n):
            c += 1
    return c

# ---- versiÃ³n numba (njit) serial ----
from numba import njit
@njit(cache=True)
def is_prime_nb(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    # usar int64 para evitar overflow (i*i)
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

@njit(cache=True)
def count_primes_nb(limit):
    c = 0
    for n in range(2, limit):
        if is_prime_nb(n):
            c += 1
    return c

# ---- versiÃ³n numba paralela con prange ----
from numba import njit, prange

@njit(parallel=True, cache=True)
def count_primes_nb_parallel(limit):
    c = 0
    # nota: reduction manual usando una variable local por iteraciÃ³n no es trivial en numba,
    # pero sumando en local y retornando al final con reduction implícita via prange es soportado
    local_sum = 0
    for n in prange(2, limit):
        if is_prime_nb(n):
            local_sum += 1
    return local_sum

# ---- helper para medir tiempo y correr escalabilidad débil ----
def run_weak_scaling(limit_base, threads_list, runs=3):
    results = []
    
    print("Warmup: compilando Numba...")
    count_primes_nb(1000)
    count_primes_nb_parallel(1000)

    # 1. Caso Base Serial (P=1, N=Limit_base)
    print(f"Midiendo caso base: P=1, N={limit_base}")
    set_num_threads(1)
    for run in range(runs):
        t0 = time.perf_counter()
        c = count_primes_nb(limit_base)
        t1 = time.perf_counter()
        results.append(("numba_serial_base", 1, c, t1-t0))

    # 2. Escalabilidad Débil (P > 1, N = Limit_base * P)
    for threads in threads_list:
        if threads == 1: # Ya medido como caso base, lo saltamos si aparece
            continue 
            
        current_limit = limit_base * threads  
        print(f"Probando: P={threads}, N={current_limit}")
        set_num_threads(threads)
        nt = get_num_threads()
        
        for run in range(runs):
            t0 = time.perf_counter()
            c = count_primes_nb_parallel(current_limit)
            t1 = time.perf_counter()
            results.append(("numba_weak_scale", nt, c, t1-t0))

    return results

if __name__ == "__main__":
    import argparse, csv, os
    
    # 1. Definición de argumentos
    parser = argparse.ArgumentParser(description="Mide la escalabilidad débil del conteo de primos con Numba.")
    parser.add_argument("--limit_base", type=int, default=400000000, help="Límite BASE N para 1 thread (default 50e6)")
    parser.add_argument("--threads", type=str, default="1,2,4,8", help="Lista de threads a probar (separadas por comas)")
    parser.add_argument("--runs", type=int, default=2, help="Número de repeticiones por configuración")
    parser.add_argument("--out", type=str, default="resultados/numba_weak_runs.csv", help="Archivo CSV de salida")
    args = parser.parse_args()

    # 2. Preparación de la lista de threads
    threads_list = [int(x) for x in args.threads.split(",") if x.strip()!='']
    limit_base = args.limit_base

    print("--- Ejecución de Escalabilidad Débil ---")
    print("Límite Base (P=1):", limit_base)
    print("Threads a probar:", threads_list)
    
    # 3. Ejecutar las mediciones
    res = run_weak_scaling(limit_base, threads_list, runs=args.runs) 

    # 4. Guardar CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant","threads","primes_count","time_s"])
        for row in res:
            writer.writerow(row)

    print("\nResultados escritos en", args.out)