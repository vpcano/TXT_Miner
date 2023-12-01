#include <iostream>
#include <algorithm>
#include <string.h>
#include <sys/time.h>
#include "cuda.h"
#include "cuda_runtime.h"
using namespace std;

#define TXT_BLOCK_SIZE 128
#define PRIME  1000000007
#define FNV_PRIME 16777619
#define OFFSET 2166136261
#define TARGET_DIFFICULTY (int) 0.5*PRIME

typedef struct {
    uint32_t prevHash;  // Hash del bloque anterior
    char text[TXT_BLOCK_SIZE];  // Texto
    unsigned int nonce;  // Nonce
    uint32_t blockHash;  // Hash del bloque (puedes ajustar la longitud según tu método de hash)
} Block;

__global__ void fnvKernel(Block* block, size_t blockSize) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int nonce = 0;
    
    unsigned int hash = OFFSET;

    do {
        printf("Trying nonce %d\n", nonce);
        // Aplica la función fnv al campo 'text' de la estructura
        for (int i = 0; i < blockSize; ++i) {
            hash ^= static_cast<unsigned int>(block->text[i]);
            hash *= FNV_PRIME;
        }
        hash %= PRIME;
    } while (hash > TARGET_DIFFICULTY && ++nonce);

    block->nonce = nonce;
    block->blockHash = hash;
}


void printBlock(Block block) {
    int len = snprintf(NULL, 0, "%d", block.nonce);
    int spaces = (21 - len) / 2;

    printf("+---------------------+\n");
    printf("|     0x%08x      |\n", block.prevHash);
    printf("|---------------------|\n| ");
    int i = 0;
    while (block.text[i] == ' ') i++;
    for (int j=0; i<strlen(block.text) && j<16; i++) {
        if (block.text[i] == '\n' || block.text[i] == '\r') {
            continue;
        }
        putchar(block.text[i]);
        ++j;
    }
    printf("... |\n| ");
    while (block.text[i] == ' ') i++;
    for (int j=0; i<strlen(block.text) && j<16; i++) {
        if (block.text[i] == '\n' || block.text[i] == '\r') {
            continue;
        }
        putchar(block.text[i]);
        ++j;
    }
    printf("... |\n|---------------------|\n");
    printf("|%*s%d%*s|\n", spaces, "", block.nonce, spaces, "");
    printf("|---------------------|\n");
    printf("|     0x%08x      |\n", block.blockHash);
    printf("+---------------------+\n");
    printf("\n\n");
}


int main(int argc, char *argv[]) {

    char fileData[TXT_BLOCK_SIZE];
    uint32_t prevBlockHash = 0;
    Block *currentBlock, *deviceBlock;

    if (argc < 2) {
        printf("Usage: %s <text file> \n", argv[0]);
        return -1;
    }

    const char* fileName = argv[1];
    FILE* file = fopen(fileName, "rb");

    if (!file) {
        fprintf(stderr, "No se pudo abrir el archivo %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    // Obtener el tamaño del archivo
    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    int n_blocks = (TXT_BLOCK_SIZE + fileSize - 1) / TXT_BLOCK_SIZE;

    for (int i=0; i < n_blocks; i++) {

        // Calcular el tamaño del bloque actual
        size_t blockSize = (i == n_blocks - 1) ? (fileSize % TXT_BLOCK_SIZE) : TXT_BLOCK_SIZE;

        // Leer el contenido del bloque actual
        memset(fileData, 0, TXT_BLOCK_SIZE);
        fread(fileData, sizeof(char), blockSize, file);

        // Crear el bloque a procesar en la memoria del Host
        currentBlock = (Block*) malloc(sizeof(Block));
        currentBlock->prevHash = prevBlockHash;
        memcpy(currentBlock->text, fileData, blockSize);
        currentBlock->nonce = 0;
        currentBlock->blockHash = 0;


        // Copiar el bloque a la memoria del Device
        cudaMalloc((void**) &deviceBlock, sizeof(Block));
        cudaMemcpy(deviceBlock, currentBlock, sizeof(Block), cudaMemcpyHostToDevice);
        
        // Lanza el kernel
        fnvKernel<<<1, 1>>>(deviceBlock, blockSize);

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