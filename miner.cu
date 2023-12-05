#include <iostream>
#include <algorithm>
#include <string.h>
#include <sys/time.h>
#include "cuda.h"
#include "cuda_runtime.h"
using namespace std;

#define TXT_BLOCK_SIZE 128
#define FNV_PRIME 16777619
#define OFFSET 2166136261
#define TARGET_DIFFICULTY 0x000000FF
#define DEFAULT_NUM_OF_THREADS 128

typedef struct {
    uint32_t prevHash;  // Hash del bloque anterior
    char text[TXT_BLOCK_SIZE];  // Texto
    uint32_t nonce;  // Nonce
    uint32_t blockHash;  // Hash del bloque (puedes ajustar la longitud según tu método de hash)
} Block;

__global__ void fnvKernel(Block* block) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nonce, hash;
    const size_t blockSize = sizeof(uint32_t) + sizeof(char)*TXT_BLOCK_SIZE;
    size_t i;
    __shared__ int foundFlag;

    if (threadId == 0) {
        foundFlag = 0;
    }

    __syncthreads();
    
    for (nonce=threadId; nonce<UINT32_MAX && !foundFlag; nonce+=blockDim.x) {
        hash = OFFSET;

        // Aplica la función fnv a la primera parte del bloque
        for (i = 0; i < blockSize; ++i) {
            hash ^= *((char*)block + i);
            hash *= FNV_PRIME;
        }

        // Hasheo de los bytes del nonce
        for (i = 0; i < sizeof(uint32_t); ++i) {
            hash ^= *((char*)&nonce + i);
            hash *= FNV_PRIME;
        }

        if (hash <= TARGET_DIFFICULTY && !foundFlag) {
            atomicExch(&foundFlag, 1);
            printf("Thread nº %d found hash 0x%08x with nonce %u\n", threadId, hash, nonce);

            block->nonce = nonce;
            block->blockHash = hash;
        }

        __syncthreads(); 
    }

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
    printf("|%*s%u%*s|\n", spaces, "", block.nonce, spaces, "");
    printf("|---------------------|\n");
    printf("|     0x%08x      |\n", block.blockHash);
    printf("+---------------------+\n");
    printf("\n\n");
}


int main(int argc, char *argv[]) {

    uint32_t prevBlockHash = 0;
    Block *currentBlock, *deviceBlock;
    struct timeval t1, t2;
    double t_total;
    int n_threads = DEFAULT_NUM_OF_THREADS, overwrite = 0, n_blocks;
    char *fileName = NULL, *outputFile = "blockchain.bin", fileData[TXT_BLOCK_SIZE];
    size_t fileSize, blockSize;
    FILE *file, *outFile;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--overwrite") == 0) {
            overwrite = 1;
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--num_of_threads") == 0) {
            if (i + 1 < argc) {
                n_threads = atoi(argv[++i]);
                if (n_threads > 0) continue;
            }
            fprintf(stderr, "Error: expected positive integer value after -n/--num_of_threads.\n");
            fprintf(stderr, "Usage: %s <text_file> [-n/--num_of_threads number_of_threads(default=%u)] [-f/--output_file output_file(default=blockchain.bin)] [-o/--overwrite] \n", argv[0], DEFAULT_NUM_OF_THREADS);
            exit(EXIT_FAILURE);
        } else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--output_file") == 0) {
            if (i + 1 < argc) {
                outputFile = argv[++i];
            } else {
                fprintf(stderr, "Error: expected file name after -f/--output_file.\n");
                fprintf(stderr, "Usage: %s <text_file> [-n/--num_of_threads number_of_threads(default=%u)] [-f/--output_file output_file(default=blockchain.bin)] [-o/--overwrite] \n", argv[0], DEFAULT_NUM_OF_THREADS);
                exit(EXIT_FAILURE);
            }
        } else {
            fileName = argv[i];
        }
    }

    if (fileName == NULL) {
        fprintf(stderr, "Error: expected input file\n");
        fprintf(stderr, "Usage: %s <text_file> [-n/--num_of_threads number_of_threads(default=%u)] [-f/--output_file output_file(default=blockchain.bin)] [-o/--overwrite] \n", argv[0], DEFAULT_NUM_OF_THREADS);
        exit(EXIT_FAILURE);
    }
    file = fopen(fileName, "rb");
    if (!file) {
        fprintf(stderr, "Error: can't open file %s\n", fileName);
        exit(EXIT_FAILURE);
    }
    // Obtener el tamaño del archivo
    fseek(file, 0, SEEK_END);
    fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    outFile = fopen(outputFile, overwrite ? "w+b" : "a+b");
    if (!outFile) {
        fprintf(stderr, "Error: can't open file %s\n", outputFile);
        exit(EXIT_FAILURE);
    }

    n_blocks = (TXT_BLOCK_SIZE + fileSize - 1) / TXT_BLOCK_SIZE;

    gettimeofday(&t1, NULL);

    for (int i=0; i < n_blocks; i++) {

        // Calcular el tamaño del bloque actual
        blockSize = (i == n_blocks - 1) ? (fileSize % TXT_BLOCK_SIZE) : TXT_BLOCK_SIZE;

        // Leer el contenido del bloque actual
        memset(fileData, 0, TXT_BLOCK_SIZE);
        fread(fileData, sizeof(char), blockSize, file);

        // Crear el bloque a procesar en la memoria del Host
        currentBlock = (Block*) malloc(sizeof(Block));
        currentBlock->prevHash = prevBlockHash;
        memcpy(currentBlock->text, fileData, blockSize);
        currentBlock->nonce = 0;
        currentBlock->blockHash = 0;


        cudaMalloc((void**) &deviceBlock, sizeof(Block));


        // Copiar el bloque a la memoria del Device
        cudaMemcpy(deviceBlock, currentBlock, sizeof(Block), cudaMemcpyHostToDevice);
        
        // Lanza el kernel
        fnvKernel<<<1, n_threads>>>(deviceBlock);

        // Copiar el bloque minado del Device al Host
        cudaMemcpy(currentBlock, deviceBlock, sizeof(Block), cudaMemcpyDeviceToHost);
        cudaFree(deviceBlock);

        printBlock(*currentBlock);

        // Check hash has been calculated correctly
        /*
        uint32_t hash = OFFSET;
        for (size_t i = 0; i < sizeof(Block) - sizeof(uint32_t); ++i) {
            hash ^= *((char*)currentBlock + i);
            hash *= FNV_PRIME;
        }
        printf("Check hash: 0x%08x\n", hash);
        */

        fwrite(currentBlock, sizeof(Block), 1, outFile);

        prevBlockHash = currentBlock->blockHash;
        free(currentBlock);

    }

    gettimeofday(&t2, NULL);
    t_total = (t2.tv_sec - t1.tv_sec)*1000000.0 + (t2.tv_usec - t1.tv_usec);
    printf("Número de hilos: %d\nTiempo total transcurrido: %f\n", n_threads, t_total);

    fclose(file);
    fclose(outFile);

    exit(EXIT_SUCCESS);

}