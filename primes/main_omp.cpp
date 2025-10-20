#include <iostream>
#include <chrono>
#include <vector>
#include <cstdint>
#include <omp.h>

bool is_prime(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    
    for (int i = 3; i * i <= n; i += 2) {
        if (n % i == 0) {
            return false;
        }
    }
    return true;
}

int main() {
    int limit = 400000000;  // Buscar primos hasta 400 millones
    int count = 0;
    
    // Iniciar medición de tiempo
    auto start = std::chrono::high_resolution_clock::now();

    // PARALLEL FOR con reducción. Cambia schedule/static,dynamic,guided y chunksize en ejecución.
    #pragma omp parallel for reduction(+:count) schedule(runtime)
    
    // Algoritmo secuencial
    for (int n = 2; n < limit; n++) {
        if (is_prime(n)) {
            count++;
        }
    }
    
    // Terminar medición de tiempo
    auto end = std::chrono::high_resolution_clock::now();
    
    // Calcular duración
    std::chrono::duration<double> duration = end - start;
    
    // Mostrar resultados
    std::cout << "=== Versión OpenMP ===" << std::endl;
    std::cout << "Rango: 2 a " << limit << std::endl;
    std::cout << "Primos encontrados: " << count << std::endl;
    std::cout << "Tiempo: " << duration.count() << " segundos" << std::endl;
    
    return 0;
}