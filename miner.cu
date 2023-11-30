#include <iostream>
#include <algorithm>
#include <sys/time.h>
#include "cuda.h"
#include "cuda_runtime.h"
using namespace std;

#define TXT_BLOCK_SIZE 128
#define PRIME  1000000007
#define FNV_PRIME 16777619
#define OFFSET 2166136261

typedef struct {
    uint32_t prevHash;  // Hash del bloque anterior
    char text[TXT_BLOCK_SIZE];  // Texto
    unsigned int nonce;  // Nonce
    uint32_t blockHash;  // Hash del bloque (puedes ajustar la longitud según tu método de hash)
} Block;

__global__ void fnvKernel(Block* block) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    
    unsigned int hash = OFFSET;

    // Aplica la función fnv al campo 'text' de la estructura
    for (int i = 0; i < TXT_BLOCK_SIZE; ++i) {
        hash ^= static_cast<unsigned int>(block->text[i]);
        hash *= FNV_PRIME;
    }
    hash %= PRIME;

    block->blockHash = hash;
}


void printBlock(Block block) {
    printf("+------------------+\n");
    printf("|    0x%x    |\n", block.prevHash);
    printf("|------------------|\n");
    printf("| %.16s |\n", block.text);
    printf("| %.13s... |\n", block.text+16);
    printf("|------------------|\n");
    printf("| %d |\n", block.nonce);
    printf("|------------------|\n");
    printf("|    0x%x    |\n", block.blockHash);
    printf("+------------------+\n");
}


int main(int argc, char *argv[]) {

    char fileData[TXT_BLOCK_SIZE];
    uint32_t prevBlockHash = 0;
    Block *currentBlock, *deviceBlock;

    printf("Hola0\n");

    if (argc < 2) {
        printf("Usage: %s <text file> \n", argv[0]);
        return -1;
    }

    printf("Hola\n");
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

    int n_blocks = (TXT_BLOCK_SIZE + fileSize - 1) / TXT_BLOCK_SIZE;

    printf("Hola1");
    for (int i=0; i < n_blocks; i++) {
        // Leer el contenido de parte del archivo a una cadena en la memoria del host
        fread(fileData, sizeof(char), TXT_BLOCK_SIZE, file + n_blocks*TXT_BLOCK_SIZE);

        // Crear el bloque a procesar en la memoria del Host
        currentBlock = (Block*) malloc(sizeof(Block));
        currentBlock->prevHash = prevBlockHash;
        memcpy(currentBlock->text, fileData, TXT_BLOCK_SIZE);
        currentBlock->nonce = 0;
        currentBlock->blockHash = 0;

        // Copiar el bloque a la memoria del Device
        cudaMalloc((void**) &deviceBlock, sizeof(Block));
        cudaMemcpy(deviceBlock, currentBlock, sizeof(Block), cudaMemcpyHostToDevice);
        
        // Lanza el kernel
        fnvKernel<<<1, 1>>>(deviceBlock);

        // Copiar el bloque minado del Device al Host
        cudaMemcpy(currentBlock, deviceBlock, sizeof(Block), cudaMemcpyDeviceToHost);
        cudaFree(deviceBlock);
        
        printBlock(*currentBlock);
        prevBlockHash = currentBlock->blockHash;
        free(currentBlock);
    }

    fclose(file);

    exit(EXIT_SUCCESS);

}