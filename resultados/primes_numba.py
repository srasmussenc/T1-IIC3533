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

# ---- helper para medir tiempo y correr variantes ----
def run_all(limit, threads_list, runs=3):
    results = []
    # Warmup JIT: compilar una vez
    print("Warmup: compilando Numba...")
    count_primes_nb(1000)
    count_primes_nb_parallel(1000)

    # Serial python (slow) -- opcional
    # t0 = time.perf_counter(); c = count_primes_py(limit); t1 = time.perf_counter()
    # results.append(("py_serial", 1, c, t1-t0))

    # Numba single-threaded (njit but without parallel)
    for run in range(runs):
        set_num_threads(1)
        t0 = time.perf_counter()
        c = count_primes_nb(limit)
        t1 = time.perf_counter()
        results.append(("numba_serial", 1, c, t1-t0))

    # Numba parallel with varying threads
    for threads in threads_list:
        set_num_threads(threads)
        # confirm actual threads used
        nt = get_num_threads()
        for run in range(runs):
            t0 = time.perf_counter()
            c = count_primes_nb_parallel(limit)
            t1 = time.perf_counter()
            results.append(("numba_parallel", nt, c, t1-t0))

    return results

if __name__ == "__main__":
    import argparse, csv
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=400000000, help="LÃ­mite (default 400e6)")
    parser.add_argument("--threads", type=str, default="1,2,4,8", help="lista de threads separadas por comas")
    parser.add_argument("--runs", type=int, default=2, help="repeticiones por config")
    parser.add_argument("--out", type=str, default="resultados/numba_runs.csv", help="archivo CSV de salida")
    args = parser.parse_args()

    threads_list = [int(x) for x in args.threads.split(",") if x.strip()!='']
    limit = args.limit

    print("Limit:", limit)
    print("Threads to test:", threads_list)
    res = run_all(limit, threads_list, runs=args.runs)

    # Guardar CSV
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant","threads","primes_count","time_s"])
        for row in res:
            writer.writerow(row)

    print("Resultados escritos en", args.out)
    for r in res:
        print(r)
