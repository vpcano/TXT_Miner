#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Definir el tamaño de las cadenas en bytes (1KB)
#define STRING_SIZE 1024

// Kernel de CUDA para copiar datos desde el archivo a la memoria global del dispositivo
__global__ void copyDataToDevice(char* deviceArray, const char* fileData, size_t fileSize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < fileSize) {
        deviceArray[tid] = fileData[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Uso: %s <nombre_del_archivo>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const char* fileName = argv[1];
    FILE* file = fopen(fileName, "r");

    if (!file) {
        fprintf(stderr, "No se pudo abrir el archivo %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    // Obtener el tamaño del archivo
    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Leer el contenido del archivo a una cadena en la memoria del host
    char* fileData = (char*)malloc(fileSize);
    fread(fileData, sizeof(char), fileSize, file);
    fclose(file);

    // Reservar memoria en la memoria global del dispositivo para los datos
    char* deviceArray;
    cudaMalloc((void**)&deviceArray, fileSize);

    // Copiar datos desde el host al dispositivo
    cudaMemcpy(deviceArray, fileData, fileSize, cudaMemcpyHostToDevice);

    // Definir la configuración de los bloques e hilos
    int blockSize = 256;
    int numBlocks = (fileSize + blockSize - 1) / blockSize;

    // Llamar al kernel de CUDA para copiar los datos
    copyDataToDevice<<<numBlocks, blockSize>>>(deviceArray, fileData, fileSize);

    // Sincronizar para asegurarse de que todas las operaciones de CUDA hayan terminado
    cudaDeviceSynchronize();

    // Liberar la memoria en el dispositivo
    cudaFree(deviceArray);

    // Liberar la memoria en el host
    free(fileData);

    return 0;
}
