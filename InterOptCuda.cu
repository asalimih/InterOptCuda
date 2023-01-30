// nvcc kernel.cu -o kernel -std=c++11 -maxrregcount 16
// sudo env "PATH=$PATH" nvcc MetaMiRs/main.cu -o main -std=c++11 -maxrregcount 63

#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <signal.h>
#include <algorithm>
#include <set>
#include <vector>
#include <map>
#include <ctime>
#include <chrono>
#include <sys/types.h>
#include <sys/stat.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>

// TODO: dodać error codes?
// TODO: wyrównać thready do WARP_SIZE?
// TODO: może jak za dużo kombinacji to dzielić cały program na batche, nie tylko kernele
// TODO: przy benchmarkach przydałby się warmup dla karty

using namespace std;

const int DEVICE_ID = 0;
//const int WARP_SIZE = 32;
const int MAX_THREADS = 1024; // TODO: zmienić na innej karcie; Maximum number of threads per block
//const int MAX_GPU_MEMORY = 1024; // in MB
const float MEMORY_MARGIN_MULTIPLIER = 1.5;

#ifdef _WIN32
bool display = false;
bool writeToFile = true;
#endif
#ifdef __linux__
bool display = false;
bool writeToFile = true;
#endif

int ALGORITHM = 0;
int COMBINATION_LENGTH = 0;
char *FILE_NAME;
char *OUTPUT_FILE_NAME;
char *ERROR_FILE_NAME;
char *META_FILE_NAME;
char *WEIGHT_FILE_NAME;
char *COMBS_FILE_NAME;
char *UNSTABLEMIRS_FILE_NAME;

#ifdef _WIN32
string LOCK_FILE_NAME = ".mirlock";
#endif
#ifdef __linux__
string LOCK_FILE_NAME = "./app/.mirlock";
#endif

int MIRS;
int SAMPLES;
int METHOD = 1;
int GEOMETRIC = 1;
int NUMOFUNSTABLEMIRS = 0;
int *groups;

//const int SAMPLE_NUMBER_THRESHOLD = 19;

string **fileData;
int **mirCombinations;
//short int *mirCombinationsFlat;
int *mirCombinationsFlat;
unsigned long long int COMBINATION_NUMBER = 1;
unsigned long long int *indeces;
unsigned long long int *differences;
unsigned long long int indecesIndex = 0;

char WeightFileName[100] = "sampletest_v1.txt";
float* WeightofMeanHostFlat = 0;

int* UnstablemiRsHostFlat = 0;

ofstream errorFile;
ofstream metaFile;

float free_m;

void writeToErrorFile(string errorMessage) {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time(&rawtime);
    timeinfo = localtime(&rawtime);
    
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);
    string currentTime(buffer);

    errorFile.open(ERROR_FILE_NAME, ios_base::app);
    errorFile << currentTime << " " << errorMessage << endl;
    errorFile.close();
}

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        cout << "CUDA Runtime Error: " << cudaGetErrorName(result) << " - " << cudaGetErrorString(result) << endl;
        writeToErrorFile("CUDA Runtime Error: " + (string)cudaGetErrorName(result) + " - " + cudaGetErrorString(result));
    }

    return result;
}

__global__ void kernelGeNorm(
    int MIRS, int SAMPLES, int COMBINATION_LENGTH, int combinationNumber, int METHOD, int GEOMETRIC,
    float *data, short int *combinations, float* cudaW, float *rankingAll, float *stabilityAll,
    float *V, float *M, float *extraMir, float *M1, float *tmpA1, float *V1, float *dataAll
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= combinationNumber) return;

    int i, j, o;

    for (j = 0; j < SAMPLES; j++) {
        extraMir[index * SAMPLES + j] = 0.0;
		if(index>70 and index<75){
			for(int i=0;i<3;i++){
			//printf("w %f c %i", cudaW[i], combinations[i]);
			}
		}
		
		if (GEOMETRIC==1){ 
			if (METHOD == 1){ //data is ct
				for (i = 0; i < COMBINATION_LENGTH; i++)
					extraMir[index * SAMPLES + j] += dataAll[combinations[index * COMBINATION_LENGTH + i] * SAMPLES + j]*cudaW[index * COMBINATION_LENGTH + i];
			} else if (METHOD == 2){ //data is cpm
				for (i = 0; i < COMBINATION_LENGTH; i++)
					extraMir[index * SAMPLES + j] += log2f(dataAll[combinations[index * COMBINATION_LENGTH + i] * SAMPLES + j])*cudaW[index * COMBINATION_LENGTH + i];
				extraMir[index * SAMPLES + j] = pow(2,extraMir[index * SAMPLES + j]);
			}
		} else {
			if (METHOD == 1){ //data is ct
				for (i = 0; i < COMBINATION_LENGTH; i++)
					extraMir[index * SAMPLES + j] += pow(2, 30-dataAll[combinations[index * COMBINATION_LENGTH + i] * SAMPLES + j])*cudaW[index * COMBINATION_LENGTH + i];
				extraMir[index * SAMPLES + j] = 30-log2f(extraMir[index * SAMPLES + j]);
			} else if (METHOD == 2){ //data is cpm
				for (i = 0; i < COMBINATION_LENGTH; i++)
					extraMir[index * SAMPLES + j] += dataAll[combinations[index * COMBINATION_LENGTH + i] * SAMPLES + j]*cudaW[index * COMBINATION_LENGTH + i];
			}
		}
        //extraMir[index * SAMPLES + j] /= COMBINATION_LENGTH;
    }
    
    float avg1;
    float sum1 = 0.0;

    M1[index * (MIRS + 1) + MIRS] = 0.0;

    for (j = 0; j < MIRS; j++) {
        avg1 = 0.0;

        for (i = 0; i < SAMPLES; i++) {
            if (METHOD == 1) {
                tmpA1[index * SAMPLES + i] = data[j * SAMPLES + i] - extraMir[index * SAMPLES + i];
            }
            else {
                tmpA1[index * SAMPLES + i] = log2f(extraMir[index * SAMPLES + i] / data[j * SAMPLES + i]);
            }

            avg1 += tmpA1[index * SAMPLES + i] / SAMPLES;
        }

        sum1 = 0.0;

        for (i = 0; i < SAMPLES; i++) {
            sum1 += (tmpA1[index * SAMPLES + i] - avg1) * (tmpA1[index * SAMPLES + i] - avg1);
        }

        V1[index * (MIRS + 1) + j] = sqrtf(sum1 / (SAMPLES - 1));
        M1[index * (MIRS + 1) + MIRS] += V1[index * (MIRS + 1) + j] / (MIRS + 1);
    }

    V1[index * (MIRS + 1) + MIRS] = 0.0;
    
    float stability = 0.0;
    float ranking = 0.0;
    float sumM1 = 0.0;

    for (j = 0; j < MIRS; j++) {
        M1[index * (MIRS + 1) + j] = M[j];
    }

    int maxIndex;
    bool extraMirRemoved;

    // if (index==0){
    //     for (i = 0; i <= MIRS; i++){
            // if (V1[i] != 0){
                // printf("%f\n", V1[i]);
            // }
    //     }
    //     printf("\n");
    // }

    for (o = 0; o < MIRS; o++) {
        extraMirRemoved = false;
        maxIndex = 0;
        sumM1 = 0.0;

        for (j = 0; j < MIRS; j++) {
            if (M1[index * (MIRS + 1) + j + 1] > M1[index * (MIRS + 1) + maxIndex]) {
                maxIndex = j + 1;
            }
        }

        for (j = 0; j < MIRS; j++) {
            sumM1 += M1[index * (MIRS + 1) + j];
        }
        sumM1 += M1[index * (MIRS + 1) + MIRS];
        stability = sumM1 / (MIRS - o + 1);

        for (j = 0; j < MIRS; j++) {
            if (M1[index * (MIRS + 1) + j] != 0.0) {
                // sumM1 += M1[index * (MIRS + 1) + j];
                if (maxIndex == MIRS) {
                    extraMirRemoved = true;
                    break;
                }
                else {
                    M1[index * (MIRS + 1) + j] = (M1[index * (MIRS + 1) + j] * (MIRS - o + 1) - V[MIRS * maxIndex + j]) / (MIRS - o);
                }
            }
        }

        if (!extraMirRemoved){
            M1[index * (MIRS + 1) + MIRS] = (M1[index * (MIRS + 1) + MIRS] * (MIRS - o + 1) - V1[index * (MIRS + 1) + maxIndex]) / (MIRS - o);
        }

        // if (M1[index * (MIRS + 1) + MIRS] != 0.0) {
        //     if (maxIndex == MIRS) {
        //         // M1[index * (MIRS + 1) + MIRS] = 0.0;
        //     }
        //     else {
                // M1[index * (MIRS + 1) + MIRS] = (M1[index * (MIRS + 1) + MIRS] * (MIRS - o + 1) - V1[index * (MIRS + 1) + maxIndex]) / (MIRS - o);
        //     }
        // }


        if (extraMirRemoved) {
            //stability = sumM1 / (MIRS - o + 1);
            ranking = MIRS - o - 1;

            break;
        }
        // else if (o == MIRS - 2) {
        //     stability = sumM1 / 2;
        //     ranking = 0.5;
        // }

        

        M1[index * (MIRS + 1) + maxIndex] = 0.0;

    }

    if (o == MIRS - 1 || ranking == 0) {
        stability = sumM1 / 2;
        // printf("%f\n", stability);
        ranking = 0.5;
    }   

    rankingAll[index] = 1.0 * ranking / MIRS;
    stabilityAll[index] = stability;
	
	// if(index>70 & index<75){
			// for(int i=0;i<3;i++){
			  // printf("stability %i %f", index, ranking);
			// }
		// }
}

__global__ void kernelGeNorm2( // with in-kernel memory allocation
    int MIRS, int SAMPLES, int COMBINATION_LENGTH, int combinationNumber, int METHOD,
    float *data, short int *combinations, float *rankingAll, float *stabilityAll,
    float *V, float *M
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= combinationNumber) return;

    int i, j, o;

    float *extraMir = (float*)malloc(SAMPLES * sizeof(float));

    for (j = 0; j < SAMPLES; j++) {
        extraMir[j] = 0.0;
        for (i = 0; i < COMBINATION_LENGTH; i++) {
            extraMir[j] += data[combinations[index * COMBINATION_LENGTH + i] * SAMPLES + j];
        }
        extraMir[j] /= COMBINATION_LENGTH;
    }
    
    float *V1 = (float*)malloc((MIRS + 1) * sizeof(float));

    float avg1;
    float sum1 = 0.0;
    float *tmpA1 = (float*)malloc(SAMPLES * sizeof(float));
    // TODO: tu problem, illegal memory access:
    float *M1 = (float*)malloc((MIRS + 1) * sizeof(float));
    if (!M1) printf("%d: malloc failed, %d %d\n", index, blockIdx.x, threadIdx.x);
    M1[MIRS] = 0.0;

    for (j = 0; j < MIRS; j++) {
        avg1 = 0.0;

        for (i = 0; i < SAMPLES; i++) {
            if (METHOD == 1) {
                tmpA1[i] = data[j * SAMPLES + i] - extraMir[i];
            }
            else {
                tmpA1[i] = log2f(extraMir[i] / data[j * SAMPLES + i]);
            }

            avg1 += tmpA1[i] / SAMPLES;
        }

        sum1 = 0.0;

        for (i = 0; i < SAMPLES; i++) {
            sum1 += (tmpA1[i] - avg1) * (tmpA1[i] - avg1);
        }

        V1[j] = sqrtf(sum1 / (SAMPLES - 1));
        M1[MIRS] += V1[j] / (MIRS + 1);
    }

    free(extraMir);
    free(tmpA1);

    V1[MIRS] = 0.0;

    float stability = 0.0;
    float ranking = 0.0;
    float sumM1 = 0.0;

    for (j = 0; j < MIRS; j++) {
        M1[j] = M[j];
    }

    int maxIndex;
    bool extraMirRemoved;

    for (o = 0; o < MIRS - 1; o++) {
        extraMirRemoved = false;
        maxIndex = 0;
        sumM1 = 0.0;

        for (j = 0; j < MIRS; j++) {
            if (M1[j + 1] > M1[maxIndex]) {
                maxIndex = j + 1;
            }
        }

        //if (index == 0) printf("%d\n", maxIndex);

        for (j = 0; j < MIRS; j++) {
            if (M1[j] != 0.0) {
                sumM1 += M1[j];

                if (maxIndex == MIRS) {
                    extraMirRemoved = true;
                    break;
                }
                else {
                    M1[j] = (M1[j] * (MIRS - o + 1) - V[MIRS * maxIndex + j]) / (MIRS - o);
                }
            }
        }

        sumM1 += M1[MIRS];

        stability = sumM1 / (MIRS - o + 1);

        if (extraMirRemoved) {
            //stability = sumM1 / (MIRS - o + 1);
            ranking = MIRS - o + 1;

            break;
        }
        else if (o == MIRS - 2) {
            stability = sumM1 / 2;
            ranking = 0.5;
        }

        //if (index == 0) printf("stab: %f\n", stability);

        if (M1[MIRS] != 0.0) {
            if (maxIndex == MIRS) {
                M1[MIRS] = 0.0;
            }
            else {
                M1[MIRS] = (M1[MIRS] * (MIRS - o + 1) - V1[maxIndex]) / (MIRS - o);
            }
        }

        M1[maxIndex] = 0.0;
    }

    free(V1);
    free(M1);

    //if (ranking == 0.0) printf("ranking: %f\n", ranking);

    rankingAll[index] = 1.0 * ranking / MIRS;
    stabilityAll[index] = stability;
}

__global__ void kernelNormFinder(
    int MIRS, int SAMPLES, int COMBINATION_LENGTH, int combinationNumber,
    float *data, float *cudaW, short int *combinations, float *rankingAll, float *stabilityAll, 
    int G, const int *groups, int *groupElements, float *sampleAvg, 
    float *groupAvg, float *dGroupAvg, float dAllAvg, float gamma, float *stabilityInit,
    float *extraMir, float *extraMirAvg, float *extraMirAvgCorr, float *sigma, int METHOD, int GEOMETRIC
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= combinationNumber) return;

    int i, j;
    
    for (j = 0; j < G; j++) {
        extraMirAvg[index * G + j] = 0.0;
    }
    
    for (j = 0; j < SAMPLES; j++) {
        
		extraMir[index * SAMPLES + j] = 0.0;
		
		if (GEOMETRIC==1){ 
			if (METHOD == 1){ //data is ct
				for (i = 0; i < COMBINATION_LENGTH; i++)
					extraMir[index * SAMPLES + j] += data[combinations[index * COMBINATION_LENGTH + i] * SAMPLES + j]*cudaW[index * COMBINATION_LENGTH + i];
			} else if (METHOD == 2){ //data is cpm
				for (i = 0; i < COMBINATION_LENGTH; i++)
					extraMir[index * SAMPLES + j] += log2f(data[combinations[index * COMBINATION_LENGTH + i] * SAMPLES + j])*cudaW[index * COMBINATION_LENGTH + i];
				extraMir[index * SAMPLES + j] = pow(2,extraMir[index * SAMPLES + j]);
			}
		} else {
			if (METHOD == 1){ //data is ct
				for (i = 0; i < COMBINATION_LENGTH; i++)
					extraMir[index * SAMPLES + j] += pow(2, 30-data[combinations[index * COMBINATION_LENGTH + i] * SAMPLES + j])*cudaW[index * COMBINATION_LENGTH + i];
				extraMir[index * SAMPLES + j] = 30-log2f(extraMir[index * SAMPLES + j]);
			} else if (METHOD == 2){ //data is cpm
				for (i = 0; i < COMBINATION_LENGTH; i++)
					extraMir[index * SAMPLES + j] += data[combinations[index * COMBINATION_LENGTH + i] * SAMPLES + j]*cudaW[index * COMBINATION_LENGTH + i];
			}
		}

        extraMirAvg[index * G + groups[j]] += (extraMir[index * SAMPLES + j] / groupElements[groups[j]]);
    }

    float extraDMirAvg = 0.0;

    for (j = 0; j < G; j++) {
        extraMirAvgCorr[index * G + j] = 0.0;
        extraDMirAvg += extraMirAvg[index * G + j] / G;
    }

    for (j = 0; j < SAMPLES; j++) {
        extraMir[index * SAMPLES + j] = extraMir[index * SAMPLES + j] - extraMirAvg[index * G + groups[j]] - sampleAvg[j] + groupAvg[groups[j]];
        extraMirAvgCorr[index * G + groups[j]] += extraMir[index * SAMPLES + j] / groupElements[groups[j]];
    }

    for (j = 0; j < G; j++) {
        sigma[index * G + j] = 0.0;
    }

    for (j = 0; j < SAMPLES; j++) {
        sigma[index * G + groups[j]] += ((extraMir[index * SAMPLES + j] - extraMirAvgCorr[index * G + groups[j]]) * (extraMir[index * SAMPLES + j] - extraMirAvgCorr[index * G + groups[j]]) / ((groupElements[groups[j]] - 1) * (1 - 2.0 / MIRS)));
    }

    float stabilityFinal = 0.0;

    if (index <2){
		// printf("index %d", index);
        // for (i = 0; i < COMBINATION_LENGTH; i++) {
            // printf("ind%d %d\n", index, combinations[index * COMBINATION_LENGTH + i]);
        // }

        for (j = 0; j < G; j++){
            // printf("ind%d %f\n", index, extraMirAvg[index * G + j] - extraDMirAvg - dGroupAvg[j] + dAllAvg);
			printf("ind%d %f\n", index, extraMirAvg[index * G + j] );
			// printf("ind%d %f\n", index, gamma);
        }
    }

    if (G > 1) {
        float w;

        for (j = 0; j < G; j++) {
            w = sigma[index * G + j] / groupElements[j];

            stabilityFinal += ((gamma * fabsf(extraMirAvg[index * G + j] - extraDMirAvg - dGroupAvg[j] + dAllAvg)) / (gamma + w) + sqrtf(w + (gamma * w) / (gamma + w))) / G;
        }
    }
    else {
        stabilityFinal = sqrtf(sigma[index * G] / SAMPLES);
    }

    stabilityAll[index] = stabilityFinal;

    for (i = 0; i < MIRS; i++) {
        if (stabilityInit[i] >= stabilityFinal) {
            rankingAll[index] = 1.0 * i / MIRS;
                
            break;
        }
    }
}

__global__ void kernelNormFinder2( // with in-kernel memory allocation
    int MIRS, int SAMPLES, int COMBINATION_LENGTH, int combinationNumber,
    float *data, short int *combinations, float *rankingAll, float *stabilityAll,
    int G, const int *groups, int *groupElements, float *sampleAvg,
    float *groupAvg, float *dGroupAvg, float dAllAvg, float gamma, float *stabilityInit
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= combinationNumber) return;

    int i, j;
    float *extraMir = (float*)malloc(SAMPLES * sizeof(float));
    float *extraMirAvg = (float*)malloc(G * sizeof(float));
    for (j = 0; j < G; j++) {
        extraMirAvg[j] = 0.0;
    }

    for (j = 0; j < SAMPLES; j++) {
        extraMir[j] = 0.0;

        for (i = 0; i < COMBINATION_LENGTH; i++) {
            extraMir[j] += data[combinations[index * COMBINATION_LENGTH + i] * SAMPLES + j];
        }

        extraMir[j] /= COMBINATION_LENGTH;
        extraMirAvg[groups[j]] += (extraMir[j] / groupElements[groups[j]]);
    }

    float *extraMirAvgCorr = (float*)malloc(G * sizeof(float));
    float extraDMirAvg = 0.0;

    for (j = 0; j < G; j++) {
        extraMirAvgCorr[j] = 0.0;
        extraDMirAvg += extraMirAvg[j] / G;
    }

    for (j = 0; j < SAMPLES; j++) {
        extraMir[j] = extraMir[j] - extraMirAvg[groups[j]] - sampleAvg[j] + groupAvg[groups[j]];
        //if (index == 1) printf("%d: %f\n", index, extraMir[j]);
        extraMirAvgCorr[groups[j]] += extraMir[j] / groupElements[groups[j]];
    }

    float *sigma = (float*)malloc(G * sizeof(float));

    for (j = 0; j < G; j++) {
        sigma[j] = 0.0;
    }

    for (j = 0; j < SAMPLES; j++) {
        sigma[groups[j]] += ((extraMir[j] - extraMirAvgCorr[groups[j]]) * (extraMir[j] - extraMirAvgCorr[groups[j]]) / ((groupElements[groups[j]] - 1) * (1 - 2.0 / MIRS)));
        //if (index == 1) printf("%d: %f\n", index, sigma[groups[j]]);
    }

    free(extraMir);
    free(extraMirAvgCorr);

    float stabilityFinal = 0.0;

    if (G > 1) {
        float w;

        for (j = 0; j < G; j++) {
            w = sigma[j] / groupElements[j];

            stabilityFinal += ((gamma * fabsf(extraMirAvg[j] - extraDMirAvg - dGroupAvg[j] + dAllAvg)) / (gamma + w) + sqrtf(w + (gamma * w) / (gamma + w))) / G;
        }
    }
    else {
        stabilityFinal = sqrtf(sigma[0] / SAMPLES);
    }

    free(extraMirAvg);
    free(sigma);

    //printf("%d: %f\n", index, stabilityFinal);

    stabilityAll[index] = stabilityFinal;

    for (i = 0; i < MIRS; i++) {
        if (stabilityInit[i] >= stabilityFinal) {
            rankingAll[index] = 1.0 * i / MIRS;

            break;
        }
    }
}

__global__ void kernelBestKeeper(
    int MIRS, int SAMPLES, int COMBINATION_LENGTH, int combinationNumber,
    float *data, float *cudaW, short int *combinations, float *rankingAll, float *stabilityAll,
    float *BKI, float BKIAvg, float BKIStd, float *stabilityInit, float *extraMir, int METHOD, int GEOMETRIC
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= combinationNumber) return;

    int i, j;

    float extraMirAvg = 0.0;
	
    for (j = 0; j < SAMPLES; j++) {
		extraMir[index * SAMPLES + j] = 0.0;
		
		if (GEOMETRIC==1){ 
			if (METHOD == 1){ //data is ct
				for (i = 0; i < COMBINATION_LENGTH; i++)
					extraMir[index * SAMPLES + j] += data[combinations[index * COMBINATION_LENGTH + i] * SAMPLES + j]*cudaW[index * COMBINATION_LENGTH + i];
			} else if (METHOD == 2){ //data is cpm
				for (i = 0; i < COMBINATION_LENGTH; i++)
					extraMir[index * SAMPLES + j] += log2f(data[combinations[index * COMBINATION_LENGTH + i] * SAMPLES + j])*cudaW[index * COMBINATION_LENGTH + i];
				extraMir[index * SAMPLES + j] = pow(2,extraMir[index * SAMPLES + j]);
			}
		} else {
			if (METHOD == 1){ //data is ct
				for (i = 0; i < COMBINATION_LENGTH; i++)
					extraMir[index * SAMPLES + j] += pow(2, 30-data[combinations[index * COMBINATION_LENGTH + i] * SAMPLES + j])*cudaW[index * COMBINATION_LENGTH + i];
				extraMir[index * SAMPLES + j] = 30-log2f(extraMir[index * SAMPLES + j]);
			} else if (METHOD == 2){ //data is cpm
				for (i = 0; i < COMBINATION_LENGTH; i++)
					extraMir[index * SAMPLES + j] += data[combinations[index * COMBINATION_LENGTH + i] * SAMPLES + j]*cudaW[index * COMBINATION_LENGTH + i];
			}
		}
		
        extraMirAvg += extraMir[index * SAMPLES + j] / SAMPLES;
		
    }
    
    float covSum = 0.0;
    float mirStd = 0.0;
    
    for (j = 0; j < SAMPLES; j++) {
        covSum += ((extraMir[index * SAMPLES + j] - extraMirAvg) * (BKI[j] - BKIAvg));
        mirStd += (extraMir[index * SAMPLES + j] - extraMirAvg) * (extraMir[index * SAMPLES + j] - extraMirAvg);
    }

    stabilityAll[index] = 1 - ((covSum / SAMPLES) / (sqrtf(mirStd / SAMPLES) * BKIStd));
	
    for (i = 0; i < MIRS; i++) {
        if (stabilityInit[i] >= stabilityAll[index]) {
            rankingAll[index] = (1.0 * i / MIRS);
            break;
        }
    }
}

__global__ void kernelBestKeeper2( // with in-kernel memory allocation
    int MIRS, int SAMPLES, int COMBINATION_LENGTH, int combinationNumber,
    float *data, short int *combinations, float *rankingAll, float *stabilityAll,
    float *BKI, float BKIAvg, float BKIStd, float *stabilityInit
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= combinationNumber) return;

    int i, j;

    float *extraMir = (float*)malloc(SAMPLES * sizeof(float));
    float extraMirAvg = 0.0;

    for (j = 0; j < SAMPLES; j++) {
        extraMir[j] = 0.0;

        for (i = 0; i < COMBINATION_LENGTH; i++) {
            extraMir[j] += data[combinations[index * COMBINATION_LENGTH + i] * SAMPLES + j];
        }

        extraMir[j] /= COMBINATION_LENGTH;
        extraMirAvg += extraMir[j] / SAMPLES;
    }

    float covSum = 0.0;
    float mirStd = 0.0;

    for (j = 0; j < SAMPLES; j++) {
        covSum += ((extraMir[j] - extraMirAvg) * (BKI[j] - BKIAvg));
        mirStd += (extraMir[j] - extraMirAvg) * (extraMir[j] - extraMirAvg);
    }

    free(extraMir);

    stabilityAll[index] = 1 - ((covSum / SAMPLES) / (sqrtf(mirStd / SAMPLES) * BKIStd));

    for (i = 0; i < MIRS; i++) {
        if (stabilityInit[i] >= stabilityAll[index]) {
            rankingAll[index] = (1.0 * i / MIRS);
            break;
        }
    }
}

void calculateCombinationNumber()
{
    for (int i = MIRS; i > (MIRS - COMBINATION_LENGTH); i--) {
        COMBINATION_NUMBER *= i;
        COMBINATION_NUMBER /= (MIRS - i + 1);
    }
}

void calculateIndeces(unsigned long long int currentIndex, int groupSize, unsigned long long int currentDifference)
{
    unsigned long long int i;

    if (indecesIndex >= COMBINATION_NUMBER) {
        return;
    }

    indeces[indecesIndex] = currentIndex;
    //cout << currentIndex << "\t";
    indecesIndex++;

    if (groupSize == 1) {
        return;
    }

    for (i = 0; (i < COMBINATION_LENGTH && (differences[i] <= currentDifference || currentDifference == 0)); i++) {
        calculateIndeces(currentIndex + differences[i], groupSize - 1, differences[i]);
    }
}

void convertIndecesToCombinations()
{
    for (int i = 0; i < COMBINATION_NUMBER; i++) {
        for (int j = (COMBINATION_LENGTH - 1); j >= 0; j--) {
            //cout << i << "\t" << j << "\t";
            mirCombinationsFlat[i * COMBINATION_LENGTH + (COMBINATION_LENGTH - 1 - j)] = (int)floor(indeces[i] / pow(MIRS, j));
            //cout << "okkkkkkk" << endl;
            indeces[i] = indeces[i] % (unsigned long long int)pow(MIRS, j);
            //cout << i << " " << j << " " << (COMBINATION_LENGTH - 1 - j) << ": " << mirCombinationsFlat[i * COMBINATION_LENGTH + (COMBINATION_LENGTH - 1 - j)] << "\t";
        }
        //cout << endl;
    }
	/*for (int i = 0; i < 200; i++) {
        for (int j = (COMBINATION_LENGTH - 1); j >= 0; j--) {
		cout<< "mirCombinationsFlat["<<i * COMBINATION_LENGTH + (COMBINATION_LENGTH - 1 - j)<<"]: "<<mirCombinationsFlat[i * COMBINATION_LENGTH + (COMBINATION_LENGTH - 1 - j)]<<endl;
		}
	}*/
}

int mainGeNorm()
{
    //// START algorithm single
    auto timeAlgorithmSingleStart = chrono::system_clock::now();
    int i, j, k;
    
    //swap data and dataextracmiR and remove unstable miRs
    int MIRSFiltered = MIRS - NUMOFUNSTABLEMIRS;
    string** fileDataFiltered = new string*[MIRSFiltered + 1];
    for (i = 0; i <= MIRSFiltered; i++) {
        fileDataFiltered[i] = new string[SAMPLES + 1];//maybe unnecessary
    }
    /*cout<< "mainGeNorm::MIRSFiltered: "<<MIRSFiltered<<endl;
    cout<< "mainGeNorm::MIRS: "<<MIRS<<endl;
    cout<< "mainGeNorm::NUMOFUNSTABLEMIRS: "<<NUMOFUNSTABLEMIRS<<endl; 
    for (i = 0; i < NUMOFUNSTABLEMIRS; i++){
        cout<<"mainGeNorm::UnstablemiR["<<i<<"]: "<<UnstablemiRsHostFlat[i]<<endl;
    }*/
    int count=0;
    cout << "NUMOFUNSTABLEMIRS: " << NUMOFUNSTABLEMIRS << endl;
    for (i = 0; i <= MIRS; i++){
       //cout<< "mainGeNorm::i: "<<i<<endl; 
       int DelFlag = 0;
       for (k = 0; k < NUMOFUNSTABLEMIRS; k++) {//algorithmic inefficient maybe need sorting
           cout<< "mainGeNorm::k: "<<k<<endl;
           if(UnstablemiRsHostFlat[k]==i){
               DelFlag = 1;
               break;
           }
       }
       if(DelFlag==0){
           //cout<< "mainGeNorm::i: "<<i<<endl;
           for (j = 0; j <= SAMPLES; j++) {
               fileDataFiltered[count][j] = fileData[i][j];
           }
           count++;
       }
        //cout<< "mainGeNorm::DelFlag: "<<DelFlag<<endl;
        //cout<< "mainGeNorm::count: "<<count<<endl;
    }
    //swap
    int MIRSORIG = MIRS;
    MIRS = MIRSFiltered;
    
    string**fileDataTemp = fileDataFiltered;
    fileDataFiltered = fileData;
    fileData = fileDataTemp;
    
    float **D = (float**)malloc(MIRS * sizeof(float*));
    for (i = 0; i < MIRS; i++) {
        D[i] = (float*)malloc(SAMPLES * sizeof(float));
    }
    float **V = (float**)malloc(MIRS * sizeof(float*));
    for (i = 0; i < MIRS; i++) {
        V[i] = (float*)malloc(MIRS * sizeof(float));
    }
    float *M = (float*)malloc(MIRS * sizeof(float));
    float *VFlat;
    float *dataFlat = (float*)malloc(MIRS * SAMPLES * sizeof(float));
    float *dataTmp = (float*)malloc(SAMPLES * sizeof(float));
    float *dataFlatAll = (float*)malloc(MIRSORIG * SAMPLES * sizeof(float));
    VFlat = (float*)malloc(MIRS * MIRS * sizeof(float));
    
    float ***A3D;
    A3D = (float***)malloc(MIRS * sizeof(float**));
    for (i = 0; i < MIRS; i++) {
        A3D[i] = (float**)malloc(MIRS * sizeof(float*));
        for (j = 0; j < MIRS; j++) {
            A3D[i][j] = (float*)malloc(SAMPLES * sizeof(float));
        }
    }

    for (k = 1; k <= MIRS; k++) {
        for (i = 1; i <= SAMPLES; i++) {
            if (fileData[k][i] != "") {
                D[k - 1][i - 1] = stof(fileData[k][i]);
            }
        }
    }

    for (i = 0; i < SAMPLES; i++) {
        for (j = 0; j < MIRS; j++) {
            for (k = 0; k < MIRS; k++) {
                if (METHOD == 1) {
                    A3D[j][k][i] = D[k][i] - D[j][i];
                }
                else {
                    A3D[j][k][i] = std::log2(D[j][i] / D[k][i]);
                }
            }
        }
    }

    float avg;
    float sum = 0.0;
    for (j = 0; j < MIRS; j++) {
        for (k = 0; k < MIRS; k++) {
            avg = 0.0;
            sum = 0.0;

            for (i = 0; i < SAMPLES; i++) {
                avg += A3D[j][k][i];
            }
            avg /= SAMPLES;

            for (i = 0; i < SAMPLES; i++) {
                sum += pow((A3D[j][k][i] - avg), 2);
            }

            V[j][k] = (sqrt(sum / (SAMPLES - 1)));
        }
    }

    for (i = 0; i < MIRS; i++) {
        for (j = 0; j < MIRS; j++) {
            VFlat[i * MIRS + j] = V[j][i];
        }
    }

    float *M1 = (float*)malloc(MIRS * sizeof(float));

    for (j = 0; j < MIRS; j++) {
        sum = 0.0;

        for (k = 0; k < MIRS; k++) {
            sum += V[j][k];
        }

        M[j] = sum / MIRS;
        M1[j] = M[j];
    }


    ofstream outFile;
    if (writeToFile) outFile.open(OUTPUT_FILE_NAME);

    if (COMBINATION_LENGTH == 1) {
        if (writeToFile) outFile << "Name\tStability\n";

        int maxIndex;
        float sumM1 = 0.0;
        float stability = 0.0;

        for (int o = 0; o < MIRS; o++) {
            maxIndex = 0;
            sumM1 = 0.0;

            for (j = 0; j < MIRS-1; j++) {
                if (M1[j + 1] > M1[maxIndex]) {
                    maxIndex = j + 1;
                }
            }
            
            if (o < MIRS-1) {
                for (j = 0; j < MIRS; j++) {
                    if (M1[j] != 0.0) {
                        sumM1 += M1[j];
                        if (o < MIRS-2){
                            M1[j] = (M1[j] * (MIRS - o) - VFlat[MIRS * maxIndex + j]) / (MIRS - o - 1);
                        }
                    }
                }

                stability = sumM1 / (MIRS - o);
            }
            
            if (display) cout << fileData[maxIndex + 1][0] << "\t" << stability << endl;
            if (writeToFile) outFile << fileData[maxIndex + 1][0] << "\t" << stability << endl;

            M1[maxIndex] = 0.0;

        }
    }
    cout << "MIRS: " << MIRS << endl;
    //print fileData
    //for (i = 1; i <= MIRS; i++) {
    //    for (j = 1; j <= SAMPLES; j++) {
    //        cout << "[" << i <<"][" << j << "]: " << stof(fileData[i][j]) << endl;
    //    }
    //} 
    //Array to control miRs in evaluation of stability
    /*float *UnstablemiRs = (float*) malloc(MIRS * sizeof(float));
    for (int o = 0; o < MIRS; o++) {
        UnstablemiRs[o] = 1.0;//weight for contribution in stability
    }*/
    auto timeAlgorithmSingleEnd = chrono::system_clock::now();
    metaFile << "time\talgorithmSingle" << "\t" << ((chrono::duration<double>)(timeAlgorithmSingleEnd - timeAlgorithmSingleStart) * 1000.0).count() << "\tms" << endl;
    //// END algorithm single
    //cout<<"Start the procdure!" << endl;
    if (COMBINATION_LENGTH > 1) {
        if (writeToFile) outFile << "Name\tRanking\tStability\n";
        //cout<< "mainGeNorm::c0"<<endl;
        for (i = 1; i <= MIRS; i++) {
            for (j = 1; j <= SAMPLES; j++) {
                dataFlat[(i-1) * SAMPLES + j-1] = stof(fileData[i][j]);
            }
        }
        //cout<< "mainGeNorm::c1"<<endl;
        //flat all of data to be used in creating combinations
        for (i = 1; i <= MIRSORIG; i++) {
            for (j = 1; j <= SAMPLES; j++) {
                dataFlatAll[(i-1) * SAMPLES + j-1] = stof(fileDataFiltered[i][j]);//fileDataFiltered is swaped with original fileData
            }
        }
        //cout<< "mainGeNorm::c2"<<endl;
        float potentialMemory = (float)(
            MIRS * SAMPLES * sizeof(float)
            + MIRS * MIRS * sizeof(float)
            + (MIRS + 1) * sizeof(float)
            + COMBINATION_NUMBER * COMBINATION_LENGTH * sizeof(short int)
            + COMBINATION_NUMBER * sizeof(float)
            + COMBINATION_NUMBER * sizeof(float)
            + COMBINATION_NUMBER * SAMPLES * sizeof(float)
            + COMBINATION_NUMBER * (MIRS + 1) * sizeof(float)
            + COMBINATION_NUMBER * SAMPLES * sizeof(float)
            + COMBINATION_NUMBER * (MIRS + 1) * sizeof(float)
        ) / 1048576.0;

        int batchNumber = 1;

        cout << "potentialMemory: " << potentialMemory << endl;

        if (potentialMemory * MEMORY_MARGIN_MULTIPLIER > free_m) {
            batchNumber = (int)ceil(potentialMemory * MEMORY_MARGIN_MULTIPLIER / free_m) + 1;
        }
        cout << "batchNumber: " << batchNumber << endl;

        int batchSize = (int)floor(COMBINATION_NUMBER / batchNumber) + 1;
        int batchSizeInit = batchSize;

        float *ranking = (float*)malloc(batchSize * sizeof(float));
        float *stabilityOut = (float*)malloc(batchSize * sizeof(float));
        short int *mirCombinationsFlatBatch = (short int*)malloc(batchSize * COMBINATION_LENGTH * sizeof(short int));
        float *WeightofMeanHostFlatBatch = (float*)malloc(batchSize * COMBINATION_LENGTH * sizeof(float));

        float *cudaRanking;
        float *cudaData;
        float *cudaV;
        float *cudaM;
        float *cudaStability;
        short int *cudaCombinations;
        float *cudaExtraMir;
        float *cudaM1;
        float *cudaTmpA1;
        float *cudaV1;
        float *cudaW;
        float *cudaDataAll; 

        //// START cuda malloc
        auto timeCudaMallocStart = chrono::system_clock::now();
        checkCuda(cudaMalloc((float**)&cudaData, MIRS * SAMPLES * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaV, MIRS * MIRS * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaM, (MIRS + 1) * sizeof(float)));
        checkCuda(cudaMalloc((short int**)&cudaCombinations, batchSize * COMBINATION_LENGTH * sizeof(short int)));
        checkCuda(cudaMalloc((float**)&cudaRanking, batchSize * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaStability, batchSize * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaExtraMir, batchSize * SAMPLES * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaM1, batchSize * (MIRS + 1) * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaTmpA1, batchSize * SAMPLES * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaV1, batchSize * (MIRS + 1) * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaW,batchSize * COMBINATION_LENGTH * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaDataAll,MIRSORIG * SAMPLES * sizeof(float)));
        auto timeCudaMallocEnd = chrono::system_clock::now();
        metaFile << "time\tcudaMalloc" << "\t" << ((chrono::duration<double>)(timeCudaMallocEnd - timeCudaMallocStart) * 1000.0).count() << "\tms" << endl;
        //// END cuda malloc

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float elapsedTimeKernel, elapsedTimeMemcpyHtD, elapsedTimeMemcpyDtH;

        for (k = 0; k < batchNumber; k++) {
            //// START batch
            auto timeBatchStart = chrono::system_clock::now();
            cout << "####################### Batch " << k << endl;

            if (k == batchNumber - 1) {
                batchSize = COMBINATION_NUMBER - k * batchSize;
            }
            cout << "batchSize: " << batchSize << endl;

            dim3 threadsPerBlock(batchSize);
            dim3 blocksPerGrid(1);

            if (batchSize > MAX_THREADS) {
                int divisor = (int)ceil((float)batchSize / MAX_THREADS);
                threadsPerBlock.x = (int)ceil(1.0 * batchSize / divisor);
                blocksPerGrid.x = divisor;
            }

            cout << "treads x: " << threadsPerBlock.x << endl;
            cout << "blocks x: " << blocksPerGrid.x << endl;

            copy(mirCombinationsFlat + COMBINATION_LENGTH * batchSizeInit * k, mirCombinationsFlat + COMBINATION_LENGTH * (batchSizeInit * k + batchSize), mirCombinationsFlatBatch);
            copy(WeightofMeanHostFlat + COMBINATION_LENGTH * batchSizeInit * k, WeightofMeanHostFlat + COMBINATION_LENGTH * (batchSizeInit * k + batchSize), WeightofMeanHostFlatBatch);

            //// START memcpy HtD
            cudaEventRecord(start, 0);
            checkCuda(cudaMemcpy(cudaData, dataFlat, MIRS * SAMPLES * sizeof(float), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(cudaM, M, (MIRS + 1) * sizeof(float), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(cudaV, VFlat, MIRS * MIRS * sizeof(float), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(cudaCombinations, mirCombinationsFlatBatch, batchSize * COMBINATION_LENGTH * sizeof(short int), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(cudaW, WeightofMeanHostFlatBatch, batchSize * COMBINATION_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(cudaRanking, ranking, batchSize * sizeof(float), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(cudaStability, stabilityOut, batchSize * sizeof(float), cudaMemcpyHostToDevice));
	        checkCuda(cudaMemcpy(cudaDataAll, dataFlatAll , MIRSORIG * SAMPLES * sizeof(float), cudaMemcpyHostToDevice));
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTimeMemcpyHtD, start, stop);
            metaFile << "time\tmemcpyHtD" << k << "\t" << elapsedTimeMemcpyHtD << "\tms" << endl;
            //// END memcpy HtD
            //cout<< "mainGeNorm::c5"<<endl;
            //// START kernel
            cudaEventRecord(start, 0);
            kernelGeNorm << <blocksPerGrid, threadsPerBlock >> > 
                (MIRS, SAMPLES, COMBINATION_LENGTH, batchSize, METHOD, GEOMETRIC,
                cudaData, cudaCombinations, cudaW, cudaRanking, cudaStability, 
                cudaV, cudaM, cudaExtraMir, cudaM1, cudaTmpA1, cudaV1, cudaDataAll);
            //kernelGeNorm2 << <blocksPerGrid, threadsPerBlock >> > (MIRS, SAMPLES, COMBINATION_LENGTH, batchSize, METHOD, cudaData, cudaCombinations, cudaRanking, cudaStability, cudaV, cudaM);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTimeKernel, start, stop);
            metaFile << "time\tkernel" << k << "\t" << elapsedTimeKernel << "\tms" << endl;
            //// END kernel
            // cout<< "mainGeNorm::c6"<<endl;
            checkCuda(cudaPeekAtLastError());

            //// START memcpy DtH
            cudaEventRecord(start, 0);
            checkCuda(cudaMemcpy(ranking, cudaRanking, batchSize * sizeof(float), cudaMemcpyDeviceToHost));
            checkCuda(cudaMemcpy(stabilityOut, cudaStability, batchSize * sizeof(float), cudaMemcpyDeviceToHost));
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTimeMemcpyDtH, start, stop);
            for(i=0;i<batchSize;i++)
                cout << i << " : " << ranking[i] << endl;
            /*for(i=0;i<batchSize;i++)
                cout << i << " : " << stabilityOut[i] << endl;*/
            metaFile << "time\tmemcpyDtH" << k << "\t" << elapsedTimeMemcpyDtH << "\tms" << endl;
            //// END memcpy DtH
            //cout<< "mainGeNorm::c7"<<endl;	
            //// START file write
            auto timeFileWriteStart = chrono::system_clock::now();
            //cout<< "mainGeNorm::c71"<<endl;
            for (i = 0; i < batchSize; i++) {
				if(i>30)
					display = false;
                //cout<< "mainGeNorm::c72"<<endl;
				if (display) cout<<i<<" ";
                for (j = 0; j < COMBINATION_LENGTH; j++) {
                    if (display) cout << fileDataFiltered[mirCombinationsFlatBatch[i * COMBINATION_LENGTH + j] + 1][0];
                    if (writeToFile) outFile << fileDataFiltered[mirCombinationsFlatBatch[i * COMBINATION_LENGTH + j] + 1][0];

                    if (j < COMBINATION_LENGTH - 1) {
                        if (display) cout << " + ";
                        if (writeToFile) outFile << " + ";
                    }
                }
                // cout<< "mainGeNorm::c73"<<endl;
                if (display) cout << "\t" << ranking[i] << "\t" << stabilityOut[i] << "\n";
                if (writeToFile) outFile << "\t" << ranking[i] << "\t" << stabilityOut[i] << "\n";
            }
            //cout<< "mainGeNorm::c74"<<endl;
            auto timeFileWriteEnd = chrono::system_clock::now();
            metaFile << "time\tfileWrite" << k << "\t" << ((chrono::duration<double>)(timeFileWriteEnd - timeFileWriteStart) * 1000.0).count() << "\tms" << endl;
            //// END file write
            //cout<< "mainGeNorm::c8"<<endl;
            auto timeBatchEnd = chrono::system_clock::now();
            metaFile << "time\tbatch" << k << "\t" << ((chrono::duration<double>)(timeBatchEnd - timeBatchStart) * 1000).count() << "\tms" << endl;
            //// END batch
        }
        //cout<< "mainGeNorm::c9"<<endl;
        cudaFree(cudaData);
        cudaFree(cudaV);
        cudaFree(cudaM);
        cudaFree(cudaCombinations);
        cudaFree(cudaW);
        cudaFree(cudaRanking);
        cudaFree(cudaStability);
        cudaFree(dataFlatAll);
        free(stabilityOut);
        free(ranking);
        free(mirCombinationsFlatBatch);
        free(WeightofMeanHostFlatBatch);

    }

    if (writeToFile) outFile.close();

    for (i = 0; i < MIRS; i++) {
        for (j = 0; j < MIRS; j++) {
            free(A3D[i][j]);
        }
        free(A3D[i]);
    }
    free(A3D);
    free(VFlat);
    //cout<< "mainGeNorm::c10"<<endl;
    return 0;
}

int mainNormFinder()
{
    //// START algorithm single
    auto timeAlgorithmSingleStart = chrono::system_clock::now();
    int i, j, k;

    float **D = (float**)malloc(MIRS * sizeof(float*));
    for (i = 0; i < MIRS; i++) {
        D[i] = (float*)malloc(SAMPLES * sizeof(float));
    }
    float *dataFlat = (float*)malloc(MIRS * SAMPLES * sizeof(float));

    for (k = 1; k <= MIRS; k++) {
        for (i = 1; i <= SAMPLES; i++) {
            if (fileData[k][i] != "") {
                if (METHOD == 1) {
                    D[k - 1][i - 1] = stof(fileData[k][i]);
                }
                else {
                    D[k - 1][i - 1] = std::log2(stof(fileData[k][i]));
                }
            }
        }
    }

    for (i = 1; i <= MIRS; i++) {
        for (j = 1; j <= SAMPLES; j++) {
            dataFlat[(i - 1) * SAMPLES + j - 1] = D[i - 1][j - 1];
        }
    }

    set<int> sa(groups, groups + SAMPLES);
    int G = (int)sa.size();
    vector<int> groupsIndices(sa.begin(), sa.end());
    cout << "Group number: " << G << endl;

    int *groupElements = (int*)calloc(G, sizeof(int));
    int *groupNumbers = (int*)calloc(G, sizeof(int));

    for (int i = 0; i < G; i++) {
        groupNumbers[groupsIndices[i]] = i;
    }

    for (int i = 0; i < SAMPLES; i++) {
        groups[i] = groupNumbers[groups[i]];
    } 

    for (int i = 0; i < SAMPLES; i++) {
        groupElements[groups[i]]++;
    }
    
    float **mirAvg = (float**)calloc(MIRS, sizeof(float*));
    for (i = 0; i < MIRS; i++) {
        mirAvg[i] = (float*)calloc(G, sizeof(float));
    }

    float *sampleAvg = (float*)malloc(SAMPLES * sizeof(float));
    float *groupAvg = (float*)malloc(G * sizeof(float));
	float highexp_n = 0;
	
    for (i = 0; i < MIRS; i++) {
        for (j = 0; j < SAMPLES; j++) {
            mirAvg[i][groups[j]] += (D[i][j] / groupElements[groups[j]]);
        }
    }

    for (j = 0; j < SAMPLES; j++) {
        sampleAvg[j] = 0.0;
		highexp_n = 0;
        for (i = 0; i < MIRS; i++) {
			if (dataFlat[i * SAMPLES + j]<=35) {
				sampleAvg[j] += dataFlat[i * SAMPLES + j];
				highexp_n += 1;
			}
        }

        sampleAvg[j] /= highexp_n;
    }


    for (j = 0; j < G; j++) {
        groupAvg[j] = 0.0;
    }

    for (j = 0; j < SAMPLES; j++) {
        groupAvg[groups[j]] += (sampleAvg[j] / groupElements[groups[j]]);
    }


    for (i = 0; i < MIRS; i++) {
        for (j = 0; j < SAMPLES; j++) {
            D[i][j] = D[i][j] - mirAvg[i][groups[j]] - sampleAvg[j] + groupAvg[groups[j]];
        }
    }  

    float *dMirAvg = (float*)malloc(MIRS * sizeof(float));
    for (i = 0; i < MIRS; i++) {
        dMirAvg[i] = 0.0;
    }
    float *dGroupAvg = (float*)malloc(G * sizeof(float));
    for (j = 0; j < G; j++) {
        dGroupAvg[j] = 0.0;
    }
    float dAllAvg = 0.0;

    for (i = 0; i < MIRS; i++) {
        for (j = 0; j < G; j++) {
            dMirAvg[i] += (mirAvg[i][j] / G);
        }
    }

    
    for (j = 0; j < G; j++) {
        for (i = 0; i < MIRS; i++) {
            dGroupAvg[j] += (mirAvg[i][j] / MIRS);
        }
    }


    for (i = 0; i < MIRS; i++) {
        dAllAvg += dMirAvg[i] / MIRS;
    }

    float **d = (float**)malloc(MIRS * sizeof(float*));
    for (i = 0; i < MIRS; i++) {
        d[i] = (float*)malloc(G * sizeof(float));
        for (j = 0; j < G; j++) {
            d[i][j] = 0.0;
        }
    }

    float sumd = 0.0;

    for (i = 0; i < MIRS; i++) {
        for (j = 0; j < G; j++) {
            d[i][j] = mirAvg[i][j] - dMirAvg[i] - dGroupAvg[j] + dAllAvg;

            sumd += (d[i][j] * d[i][j]);
        }
    }

    float **newMirAvg = (float**)malloc(MIRS * sizeof(float*));
    for (i = 0; i < MIRS; i++) {
        newMirAvg[i] = (float*)malloc(G * sizeof(float));
        for (j = 0; j < G; j++) {
            newMirAvg[i][j] = 0.0;
        }
    }

    for (i = 0; i < MIRS; i++) {
        for (j = 0; j < SAMPLES; j++) {
            newMirAvg[i][groups[j]] += (D[i][j] / groupElements[groups[j]]);
        }
    }

    float **sigma = (float**)malloc(MIRS * sizeof(float*));
    for (i = 0; i < MIRS; i++) {
        sigma[i] = (float*)malloc(G * sizeof(float));
        for (j = 0; j < G; j++) {
            sigma[i][j] = 0.0;
        }
    }

    for (i = 0; i < MIRS; i++) {
        for (j = 0; j < SAMPLES; j++) {
            sigma[i][groups[j]] += (float)(pow(D[i][j] - newMirAvg[i][groups[j]], 2) / ((groupElements[groups[j]] - 1) * (1 - 2.0 / MIRS)));
        }
    }

    float sumSigma = 0.0;

    for (i = 0; i < MIRS; i++) {
        for (j = 0; j < G; j++) {
            sumSigma += (sigma[i][j] / groupElements[groupsIndices[j]]);
        }
    }

    float gamma = 0.0;
    float w = 0.0;

    float **stability = (float**)malloc(MIRS * sizeof(float*));
    for (i = 0; i < MIRS; i++) {
        stability[i] = (float*)malloc(G * sizeof(float));
        for (j = 0; j < G; j++) {
            stability[i][j] = 0.0;
        }
    }
    float *stabilityFinal = (float*)malloc(MIRS * sizeof(float));
    for (i = 0; i < MIRS; i++) {
        stabilityFinal[i] = 0.0;
    }

    if (G > 1) {
        gamma = (sumd / (MIRS - 1) / (G - 1)) - (sumSigma / MIRS / G);

        if (gamma < 0.0) {
            gamma = 0.0;
        }

        for (i = 0; i < MIRS; i++) {
            for (j = 0; j < G; j++) {
                w = sigma[i][j] / groupElements[groupsIndices[j]];

                stability[i][j] = (gamma * abs(d[i][j])) / (gamma + w);

                float underSqrt = w + (gamma * w) / (gamma + w);

                stability[i][j] += sqrt(underSqrt);
                stabilityFinal[i] += stability[i][j] / G;

            }
        }
    }
    else {
        for (i = 0; i < MIRS; i++) {
            stabilityFinal[i] = sqrt(sigma[i][0] / SAMPLES);  
        }
    }

    ofstream outFile;

    if (writeToFile) outFile.open(OUTPUT_FILE_NAME);
    
    if (COMBINATION_LENGTH == 1) {
        if (writeToFile) outFile << "Name\tStability\n";

        for (i = 0; i < MIRS; i++) {
            if (display) cout << fileData[i + 1][0] << "\t" << stabilityFinal[i] << endl;
            if (writeToFile) outFile << fileData[i + 1][0] << "\t" << stabilityFinal[i] << endl;
        }
    }

    auto timeAlgorithmSingleEnd = chrono::system_clock::now();
    metaFile << "time\talgorithmSingle" << "\t" << ((chrono::duration<double>)(timeAlgorithmSingleEnd - timeAlgorithmSingleStart) * 1000.0).count() << "\tms" << endl;
    //// END algorithm single

    if (COMBINATION_LENGTH > 1) {
        if (writeToFile) outFile << "Name\tRanking\tStability\n";

        float *stabilityInit = (float*)malloc(MIRS * sizeof(float));
        float *rankingInit = (float*)malloc(MIRS * sizeof(float));

        int minIndex;

        for (i = 0; i < MIRS; i++) {
            minIndex = 0;

            for (j = 0; j < MIRS; j++) {
                if (stabilityFinal[j] < stabilityFinal[minIndex]) {
                    minIndex = j;
                }
            }

            stabilityInit[i] = stabilityFinal[minIndex];
            //cout << stabilityInit[i] << endl;
            rankingInit[i] = (float)minIndex;

            stabilityFinal[minIndex] = 100000.0;
        }

        float potentialMemory = (float)(
            MIRS * SAMPLES * sizeof(float)
            + COMBINATION_NUMBER * COMBINATION_LENGTH * sizeof(short int)
            + G * sizeof(int)
            + SAMPLES * sizeof(int)
            + COMBINATION_NUMBER * sizeof(float)
            + COMBINATION_NUMBER * sizeof(float)
            + SAMPLES * sizeof(float)
            + 2 * G * sizeof(float)
            + MIRS * sizeof(float)
            + COMBINATION_NUMBER * SAMPLES * sizeof(float)
            + 3 * COMBINATION_NUMBER * G * sizeof(float)
        ) / 1048576.0;

        int batchNumber = 1;

        cout << "potentialMemory: " << potentialMemory << " MB" << endl;

        if (potentialMemory * MEMORY_MARGIN_MULTIPLIER > free_m) {
            batchNumber = (int)ceil(potentialMemory * MEMORY_MARGIN_MULTIPLIER / free_m) + 1;
        }
        cout << "batchNumber: " << batchNumber << endl;

        int batchSize = (int)floor(COMBINATION_NUMBER / batchNumber) + 1;
        int batchSizeInit = batchSize;

        float *ranking = (float*)calloc(batchSize, sizeof(float));
        float *stabilityOut = (float*)calloc(batchSize, sizeof(float));
        short int *mirCombinationsFlatBatch = (short int*)malloc(batchSize * COMBINATION_LENGTH * sizeof(short int));
		float *WeightofMeanHostFlatBatch = (float*)malloc(batchSize * COMBINATION_LENGTH * sizeof(float));
		
        float *cudaData;
        short int *cudaCombinations;
        int *cudaGroupElements;
        int *cudaGroups;
        float *cudaRanking;
        float *cudaStability;
        float *cudaSampleAvg;
        float *cudaGroupAvg;
        float *cudaDGroupAvg;
        float *cudaStabilityInit;
        float *cudaExtraMir;
        float *cudaExtraMirAvg;
        float *cudaExtraMirAvgCorr;
        float *cudaSigma;
		float *cudaW;

        //// START cuda malloc
        auto timeCudaMallocStart = chrono::system_clock::now();
        checkCuda(cudaMalloc((float**)&cudaData, MIRS * SAMPLES * sizeof(float)));
        checkCuda(cudaMalloc((short int**)&cudaCombinations, batchSize * COMBINATION_LENGTH * sizeof(short int)));
        checkCuda(cudaMalloc((int**)&cudaGroupElements, G * sizeof(int)));
        checkCuda(cudaMalloc((int**)&cudaGroups, SAMPLES * sizeof(int)));
        checkCuda(cudaMalloc((float**)&cudaRanking, batchSize * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaStability, batchSize * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaSampleAvg, SAMPLES * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaGroupAvg, G * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaDGroupAvg, G * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaStabilityInit, MIRS * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaExtraMir, batchSize * SAMPLES * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaExtraMirAvg, batchSize * G * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaExtraMirAvgCorr, batchSize * G * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaSigma, batchSize * G * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaW,batchSize * COMBINATION_LENGTH * sizeof(float)));
		
		
        auto timeCudaMallocEnd = chrono::system_clock::now();
        metaFile << "time\tcudaMalloc" << "\t" << ((chrono::duration<double>)(timeCudaMallocEnd - timeCudaMallocStart) * 1000.0).count() << "\tms" << endl;
        //// END cuda malloc

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float elapsedTimeKernel, elapsedTimeMemcpyHtD, elapsedTimeMemcpyDtH;

        for (k = 0; k < batchNumber; k++) {
            //// START batch
            auto timeBatchStart = chrono::system_clock::now();
            cout << "####################### Batch " << k << endl;

            if (k == batchNumber - 1) {
                batchSize = COMBINATION_NUMBER - k * batchSize;
            }
            cout << "batchSize: " << batchSize << endl;

            dim3 threadsPerBlock(batchSize);
            dim3 blocksPerGrid(1);

            if (batchSize > MAX_THREADS) {
                int divisor = (int)ceil((float)batchSize / MAX_THREADS);
                threadsPerBlock.x = (int)ceil(1.0 * batchSize / divisor);
                blocksPerGrid.x = divisor;
            }
            /*if (threadsPerBlock.x % WARP_SIZE != 0) {
                threadsPerBlock.x -= (threadsPerBlock.x % WARP_SIZE);
                blocksPerGrid.x += 1;
            }*/

            cout << "treads x: " << threadsPerBlock.x << endl;
            cout << "blocks x: " << blocksPerGrid.x << endl;

            copy(mirCombinationsFlat + COMBINATION_LENGTH * batchSizeInit * k, mirCombinationsFlat + COMBINATION_LENGTH * (batchSizeInit * k + batchSize), mirCombinationsFlatBatch);
            copy(WeightofMeanHostFlat + COMBINATION_LENGTH * batchSizeInit * k, WeightofMeanHostFlat + COMBINATION_LENGTH * (batchSizeInit * k + batchSize), WeightofMeanHostFlatBatch);
			
            //// START memcpy HtD
            cudaEventRecord(start, 0);
            checkCuda(cudaMemcpy(cudaData, dataFlat, MIRS * SAMPLES * sizeof(float), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(cudaCombinations, mirCombinationsFlatBatch, batchSize * COMBINATION_LENGTH * sizeof(short int), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(cudaGroupElements, groupElements, G * sizeof(int), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(cudaGroups, groups, SAMPLES * sizeof(int), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(cudaRanking, ranking, batchSize * sizeof(float), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(cudaStability, stabilityOut, batchSize * sizeof(float), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(cudaSampleAvg, sampleAvg, SAMPLES * sizeof(float), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(cudaGroupAvg, groupAvg, G * sizeof(float), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(cudaDGroupAvg, dGroupAvg, G * sizeof(float), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(cudaStabilityInit, stabilityInit, MIRS * sizeof(float), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(cudaW, WeightofMeanHostFlatBatch, batchSize * COMBINATION_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
			
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTimeMemcpyHtD, start, stop);
            metaFile << "time\tmemcpyHtD" << k << "\t" << elapsedTimeMemcpyHtD << "\tms" << endl;
            //// END memcpy HtD

            //// START kernel
            cudaEventRecord(start, 0);
            kernelNormFinder << <blocksPerGrid, threadsPerBlock >> > 
                (MIRS, SAMPLES, COMBINATION_LENGTH, batchSize, 
                cudaData, cudaW, cudaCombinations, cudaRanking, cudaStability, 
                G, cudaGroups, cudaGroupElements, cudaSampleAvg, 
                cudaGroupAvg, cudaDGroupAvg, dAllAvg, gamma, cudaStabilityInit, 
                cudaExtraMir, cudaExtraMirAvg, cudaExtraMirAvgCorr, cudaSigma, METHOD, GEOMETRIC);
            //kernelNormFinder2 << <blocksPerGrid, threadsPerBlock >> > (MIRS, SAMPLES, COMBINATION_LENGTH, batchSize, cudaData, cudaCombinations, cudaRanking, cudaStability, G, cudaGroups, cudaGroupElements, cudaSampleAvg, cudaGroupAvg, cudaDGroupAvg, dAllAvg, gamma, cudaStabilityInit);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTimeKernel, start, stop);
            metaFile << "time\tkernel" << k << "\t" << elapsedTimeKernel << "\tms" << endl;
            //// END kernel

            checkCuda(cudaPeekAtLastError());

            //// START memcpy DtH
            cudaEventRecord(start, 0);
            checkCuda(cudaMemcpy(ranking, cudaRanking, batchSize * sizeof(float), cudaMemcpyDeviceToHost));
            checkCuda(cudaMemcpy(stabilityOut, cudaStability, batchSize * sizeof(float), cudaMemcpyDeviceToHost));
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTimeMemcpyDtH, start, stop);
            metaFile << "time\tmemcpyDtH" << k << "\t" << elapsedTimeMemcpyDtH << "\tms" << endl;
            //// END memcpy DtH

            //// START file write
            auto timeFileWriteStart = chrono::system_clock::now();
            for (i = 0; i < batchSize; i++) {
                //if (k < batchNumber - 1 || i < batchSize - 200) continue;
				if(i>30)
					display = false;
				if (display) cout<<i<<" ";
                for (j = 0; j < COMBINATION_LENGTH; j++) {
                    if (display) cout << fileData[mirCombinationsFlatBatch[i * COMBINATION_LENGTH + j] + 1][0];
                    if (writeToFile) outFile << fileData[mirCombinationsFlatBatch[i * COMBINATION_LENGTH + j] + 1][0];

                    if (j < COMBINATION_LENGTH - 1) {
                        if (display) cout << " + ";
                        if (writeToFile) outFile << " + ";
                    }
                }

                if (display) cout << "\t" << ranking[i] << "\t" << stabilityOut[i] << "\n";
                if (writeToFile) outFile << "\t" << ranking[i] << "\t" << stabilityOut[i] << "\n";
            }
            auto timeFileWriteEnd = chrono::system_clock::now();
            metaFile << "time\tfileWrite" << k << "\t" << ((chrono::duration<double>)(timeFileWriteEnd - timeFileWriteStart) * 1000.0).count() << "\tms" << endl;
            //// END file write

            auto timeBatchEnd = chrono::system_clock::now();
            metaFile << "time\tbatch" << k << "\t" << ((chrono::duration<double>)(timeBatchEnd - timeBatchStart) * 1000).count() << "\tms" << endl;
            //// END batch
        }

        if (writeToFile) outFile.close();

        free(ranking);
        free(stabilityOut);
        free(mirCombinationsFlatBatch);
        free(WeightofMeanHostFlatBatch);
		
        cudaFree(cudaData);
        cudaFree(cudaCombinations);
        cudaFree(cudaGroupElements);
        cudaFree(cudaGroups);
        cudaFree(cudaRanking);
        cudaFree(cudaStability);
        cudaFree(cudaSampleAvg);
        cudaFree(cudaGroupAvg);
        cudaFree(cudaDGroupAvg);
        cudaFree(cudaStabilityInit);
        cudaFree(cudaW);
    }

    return 0;
}

int mainBestKeeper()
{
    //// START algorithm single
    auto timeAlgorithmSingleStart = chrono::system_clock::now();
    int i, j, k;

    float **D = (float**)malloc(MIRS * sizeof(float*));
    for (i = 0; i < MIRS; i++) {
        D[i] = (float*)malloc(SAMPLES * sizeof(float));
    }
    float *dataFlat = (float*)malloc(MIRS * SAMPLES * sizeof(float));
    float *BKI = (float*)malloc(SAMPLES * sizeof(float));

    for (j = 0; j < SAMPLES; j++) {
        BKI[j] = 1.0;
    }

    float allAvg = 0.0;

    for (k = 1; k <= MIRS; k++) {
        for (i = 1; i <= SAMPLES; i++) {
            if (fileData[k][i] != "") {
                if (METHOD == 1) {
                    D[k - 1][i - 1] = stof(fileData[k][i]);
                }
                else {
                    D[k - 1][i - 1] = std::log2(stof(fileData[k][i]));
                }
            }
        }
    }

    float *mirAvg = (float*)calloc(MIRS, sizeof(float));

    for (i = 0; i < MIRS; i++) {
        for (j = 0; j < SAMPLES; j++) {
            dataFlat[i * SAMPLES + j] = D[i][j];
            allAvg += D[i][j] / (MIRS * SAMPLES);
            mirAvg[i] += D[i][j] / SAMPLES;
        }
    }
    
    float BKIAvg = 0.0;

    for (j = 0; j < SAMPLES; j++) {
        for (i = 0; i < MIRS; i++) {
            BKI[j] *= (D[i][j] / allAvg);
        }

        BKI[j] = (float)pow(BKI[j], 1.0 / MIRS) * allAvg;
        BKIAvg += BKI[j] / SAMPLES;
    }

    float BKIStd = 0.0;

    for (j = 0; j < SAMPLES; j++) {
        BKIStd += pow(BKI[j] - BKIAvg, 2);
    }
    BKIStd /= SAMPLES;
    BKIStd = sqrt(BKIStd);

    float *stabilityFinal = (float*)malloc(MIRS * sizeof(float));

    float covSum;
    float mirStd;
    float r;

    ofstream outFile;
    if (writeToFile) outFile.open(OUTPUT_FILE_NAME);

    if (COMBINATION_LENGTH == 1) {
        if (writeToFile) outFile << "Name\tStability\n";
    }

    for (i = 0; i < MIRS; i++) {
        covSum = 0.0;
        mirStd = 0.0;

        for (j = 0; j < SAMPLES; j++) {
            covSum += ((D[i][j] - mirAvg[i]) * (BKI[j] - BKIAvg));
            mirStd += (D[i][j] - mirAvg[i]) * (D[i][j] - mirAvg[i]);
        }

        mirStd /= SAMPLES;
        mirStd = sqrtf(mirStd);

        r = (covSum / SAMPLES) / (mirStd * BKIStd);

        stabilityFinal[i] = 1 - r;
        
        if (COMBINATION_LENGTH == 1) {
            if (display) cout << fileData[i + 1][0] << "\t" << stabilityFinal[i] << endl;
            if (writeToFile) outFile << fileData[i + 1][0] << "\t" << stabilityFinal[i] << endl;
        }
    }

    auto timeAlgorithmSingleEnd = chrono::system_clock::now();
    metaFile << "time\talgorithmSingle" << "\t" << ((chrono::duration<double>)(timeAlgorithmSingleEnd - timeAlgorithmSingleStart) * 1000.0).count() << "\tms" << endl;
    //// END algorithm single

    if (COMBINATION_LENGTH > 1) {
        if (writeToFile) outFile << "Name\tRanking\tStability\n";

        float *stabilityInit = (float*)malloc(MIRS * sizeof(float));
        int minIndex;

        for (i = 0; i < MIRS; i++) {
            minIndex = 0;

            for (j = 0; j < MIRS; j++) {
                if (stabilityFinal[j] < stabilityFinal[minIndex]) {
                    minIndex = j;
                }
            }

            stabilityInit[i] = stabilityFinal[minIndex];
            stabilityFinal[minIndex] = 100000.0;
        }

        float potentialMemory = (float)(
            MIRS * SAMPLES * sizeof(float)
            + COMBINATION_NUMBER * COMBINATION_LENGTH * sizeof(short int)
            + COMBINATION_NUMBER * sizeof(float)
            + COMBINATION_NUMBER * sizeof(float)
            + SAMPLES * sizeof(float)
            + MIRS * sizeof(float)
            + COMBINATION_NUMBER * SAMPLES * sizeof(float)
        ) / 1048576.0;

        int batchNumber = 1;

        cout << "potentialMemory: " << potentialMemory << endl;

        if (potentialMemory * MEMORY_MARGIN_MULTIPLIER > free_m) {
            batchNumber = (int)ceil(potentialMemory * MEMORY_MARGIN_MULTIPLIER / free_m) + 1;
        }
        cout << "batchNumber: " << batchNumber << endl;

        int batchSize = (int)floor(COMBINATION_NUMBER / batchNumber) + 1;
        int batchSizeInit = batchSize;
    
        float *ranking = (float*)calloc(batchSize, sizeof(float));
        float *stabilityOut = (float*)calloc(batchSize, sizeof(float));
        short int *mirCombinationsFlatBatch = (short int*)malloc(batchSize * COMBINATION_LENGTH * sizeof(short int));
        float *WeightofMeanHostFlatBatch = (float*)malloc(batchSize * COMBINATION_LENGTH * sizeof(float));


        float *cudaData;
        short int *cudaCombinations;
        float *cudaRanking;
        float *cudaStability;
        float *cudaBKI;
        float *cudaStabilityInit;
        float *cudaExtraMir;
        float *cudaW;

        //// START cuda malloc
        auto timeCudaMallocStart = chrono::system_clock::now();
        checkCuda(cudaMalloc((float**)&cudaData, MIRS * SAMPLES * sizeof(float)));
        checkCuda(cudaMalloc((short int**)&cudaCombinations, batchSize * COMBINATION_LENGTH * sizeof(short int)));
        checkCuda(cudaMalloc((float**)&cudaRanking, batchSize * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaStability, batchSize * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaBKI, SAMPLES * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaStabilityInit, MIRS * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaExtraMir, batchSize * SAMPLES * sizeof(float)));
        checkCuda(cudaMalloc((float**)&cudaW,batchSize * COMBINATION_LENGTH * sizeof(float)));

        auto timeCudaMallocEnd = chrono::system_clock::now();
        metaFile << "time\tcudaMalloc" << "\t" << ((chrono::duration<double>)(timeCudaMallocEnd - timeCudaMallocStart) * 1000.0).count() << "\tms" << endl;
        //// END cuda malloc

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float elapsedTimeKernel, elapsedTimeMemcpyHtD, elapsedTimeMemcpyDtH;

        for (k = 0; k < batchNumber; k++) {
            //// START batch
            auto timeBatchStart = chrono::system_clock::now();
            cout << "####################### Batch " << k << endl;

            if (k == batchNumber - 1) {
                batchSize = COMBINATION_NUMBER - k * batchSize;
            }
            cout << "batchSize: " << batchSize << endl;

            dim3 threadsPerBlock(batchSize);
            dim3 blocksPerGrid(1);

            if (batchSize > MAX_THREADS) {
                int divisor = (int)ceil((float)batchSize / MAX_THREADS);
                threadsPerBlock.x = (int)ceil(1.0 * batchSize / divisor);
                blocksPerGrid.x = divisor;   
            }
            /*if (threadsPerBlock.x % WARP_SIZE != 0) {
                threadsPerBlock.x -= (threadsPerBlock.x % WARP_SIZE);
                blocksPerGrid.x += 1;
            }*/

            cout << "treads x: " << threadsPerBlock.x << endl;
            cout << "blocks x: " << blocksPerGrid.x << endl;

            copy(mirCombinationsFlat + COMBINATION_LENGTH * batchSizeInit * k, mirCombinationsFlat + COMBINATION_LENGTH * (batchSizeInit * k + batchSize), mirCombinationsFlatBatch);
			copy(WeightofMeanHostFlat + COMBINATION_LENGTH * batchSizeInit * k, WeightofMeanHostFlat + COMBINATION_LENGTH * (batchSizeInit * k + batchSize), WeightofMeanHostFlatBatch);

            //// START memcpy HtD
            cudaEventRecord(start, 0);
            checkCuda(cudaMemcpy(cudaData, dataFlat, MIRS * SAMPLES * sizeof(float), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(cudaCombinations, mirCombinationsFlatBatch, batchSize * COMBINATION_LENGTH * sizeof(short int), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(cudaRanking, ranking, batchSize * sizeof(float), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(cudaStability, stabilityOut, batchSize * sizeof(float), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(cudaBKI, BKI, SAMPLES * sizeof(float), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(cudaStabilityInit, stabilityInit, MIRS * sizeof(float), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(cudaW, WeightofMeanHostFlatBatch, batchSize * COMBINATION_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
			
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTimeMemcpyHtD, start, stop);
            metaFile << "time\tmemcpyHtD" << k << "\t" << elapsedTimeMemcpyHtD << "\tms" << endl;
            //// END memcpy HtD

            //// START kernel
            cudaEventRecord(start, 0);
            kernelBestKeeper << <blocksPerGrid, threadsPerBlock >> >
				(MIRS, SAMPLES, COMBINATION_LENGTH, batchSize, cudaData, cudaW,
				cudaCombinations,cudaRanking, cudaStability, cudaBKI, BKIAvg, BKIStd,
				cudaStabilityInit, cudaExtraMir, METHOD, GEOMETRIC);
            //kernelBestKeeper2 << <blocksPerGrid, threadsPerBlock >> > (MIRS, SAMPLES, COMBINATION_LENGTH, batchSize, cudaData, cudaCombinations, cudaRanking, cudaStability, cudaBKI, BKIAvg, BKIStd, cudaStabilityInit);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTimeKernel, start, stop);
            metaFile << "time\tkernel" << k << "\t" << elapsedTimeKernel << "\tms" << endl;
            //// END kernel

            checkCuda(cudaPeekAtLastError());

            //// START memcpy DtH
            cudaEventRecord(start, 0);
            checkCuda(cudaMemcpy(ranking, cudaRanking, batchSize * sizeof(float), cudaMemcpyDeviceToHost));
            checkCuda(cudaMemcpy(stabilityOut, cudaStability, batchSize * sizeof(float), cudaMemcpyDeviceToHost));
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTimeMemcpyDtH, start, stop);
            metaFile << "time\tmemcpyDtH" << k << "\t" << elapsedTimeMemcpyDtH << "\tms" << endl;
            //// END memcpy DtH

            //// START file write
            auto timeFileWriteStart = chrono::system_clock::now();
            for (i = 0; i < batchSize; i++) {
				if(i>30)
					display = false;
				if (display) cout<<i<<" ";
                for (j = 0; j < COMBINATION_LENGTH; j++) {
                    if (display) cout << fileData[mirCombinationsFlatBatch[i * COMBINATION_LENGTH + j] + 1][0];
                    if (writeToFile) outFile << fileData[mirCombinationsFlatBatch[i * COMBINATION_LENGTH + j] + 1][0];

                    if (j < COMBINATION_LENGTH - 1) {
                        if (display) cout << " + ";
                        if (writeToFile) outFile << " + ";
                    }
                }

                if (display) cout  << "\t" << ranking[i] << "\t" << stabilityOut[i] << "\n";
                if (writeToFile) outFile << "\t" << ranking[i] << "\t" << stabilityOut[i] << "\n";
            }
            auto timeFileWriteEnd = chrono::system_clock::now();
            metaFile << "time\tfileWrite" << k << "\t" << ((chrono::duration<double>)(timeFileWriteEnd - timeFileWriteStart) * 1000.0).count() << "\tms" << endl;
            //// END file write

            auto timeBatchEnd = chrono::system_clock::now();
            metaFile << "time\tbatch" << k << "\t" << ((chrono::duration<double>)(timeBatchEnd - timeBatchStart) * 1000).count() << "\tms" << endl;
            //// END batch
        }

        cudaFree(cudaData);
        cudaFree(cudaCombinations);
        cudaFree(cudaRanking);
        cudaFree(cudaStability);
        cudaFree(cudaBKI);
        cudaFree(cudaStabilityInit);
        cudaFree(cudaExtraMir);
	cudaFree(cudaW);
        
	free(mirCombinationsFlatBatch);
	free(WeightofMeanHostFlatBatch);
        free(ranking);
        free(stabilityOut);
        free(stabilityInit);
    }
    
    if (writeToFile) outFile.close();

    for (i = 0; i < MIRS; i++) {
        free(D[i]);
    }
    free(D);
    free(dataFlat);
    free(BKI);
    free(mirAvg);
    free(stabilityFinal);

    return 0;
}
bool ReadUnstablemiRsFile(){
    char buffer[2];
    int mirCountRow = 0;

    FILE *file;
    file = fopen(UNSTABLEMIRS_FILE_NAME, "r");
    string CurrStringTmp = "";
    if (file) {
        auto timeFileReadStart = chrono::system_clock::now();
        while (!feof(file)) {
            if (fgets(buffer, 2, file) == NULL) break;

            if (*buffer != ' ' && *buffer != '\n') {
                CurrStringTmp.append(buffer);
            }else{
            //if (*buffer == ' ') {
                UnstablemiRsHostFlat[mirCountRow] = stoi(CurrStringTmp);
                mirCountRow++;
                CurrStringTmp = "";
            }
	    //if(mirCountRow>NUMOFUNSTABLEMIRS) break;
        }

        // MIRS -= 2;
        // SAMPLES -= 1;

        fclose(file);

        auto timeFileReadEnd = chrono::system_clock::now();
        //metaFile << "time\tfileread\t" << ((chrono::duration<double>)(timeFileReadEnd - timeFileReadStart) * 1000).count() << "\tms" << endl;

        return true;
    } else {
        return false;
    }
}

bool ReadWeightFile(){
    char buffer[2];
    int mirCountRow = 0;
    int mirCountCol = 0;

    FILE *file;
    file = fopen(WEIGHT_FILE_NAME, "r");
    string CurrStringTmp = "";
    if (file) {
        auto timeFileReadStart = chrono::system_clock::now();

        while (!feof(file)) {

            if (fgets(buffer, 2, file) == NULL) break;

            if (*buffer != ' ' && *buffer != '\n') {
                CurrStringTmp.append(buffer);
            }
            
            if (*buffer == ' ') {
                WeightofMeanHostFlat[mirCountRow] = stof(CurrStringTmp);
                mirCountCol++;
                CurrStringTmp = "";
            }

            if (*buffer == '\n') {
                WeightofMeanHostFlat[mirCountRow] = stof(CurrStringTmp);
                mirCountRow++;
                mirCountCol = 0;
                CurrStringTmp = "";
            }
        }

        // MIRS -= 2;
        // SAMPLES -= 1;

        fclose(file);

        auto timeFileReadEnd = chrono::system_clock::now();
        //metaFile << "time\tfileread\t" << ((chrono::duration<double>)(timeFileReadEnd - timeFileReadStart) * 1000).count() << "\tms" << endl;

        return true;
    } else {
        return false;
    }
}


bool ReadCombsFile(){
    char buffer[2];
    int mirCountRow = 0;
    int mirCountCol = 0;

    FILE *file;
    file = fopen(COMBS_FILE_NAME, "r");
    cout << COMBS_FILE_NAME << "***" <<endl;
    string CurrStringTmp = "";
    if (file) {
        auto timeFileReadStart = chrono::system_clock::now();

        while (!feof(file)) {

            if (fgets(buffer, 2, file) == NULL) break;

            if (*buffer != ' ' && *buffer != '\n') {
                CurrStringTmp.append(buffer);
            }
            
            if (*buffer == ' ') {
				
                mirCombinationsFlat[mirCountRow] = stoi(CurrStringTmp);//stoi(CurrStringTmp);
                mirCountCol++;
                CurrStringTmp = "";
            }

            if (*buffer == '\n') {
                //cout << stoi(CurrStringTmp);
                mirCombinationsFlat[mirCountRow] = stoi(CurrStringTmp);//stoi(CurrStringTmp);
                mirCountRow++;
                mirCountCol = 0;
                CurrStringTmp = "";
                //cout << "dd"<< typeid(mirCombinationsFlat).name() << mirCombinationsFlat[mirCountRow] << endl;
            }

        }
			
        // MIRS -= 2;
        // SAMPLES -= 1;

        fclose(file);

        auto timeFileReadEnd = chrono::system_clock::now();
        //metaFile << "time\tfileread\t" << ((chrono::duration<double>)(timeFileReadEnd - timeFileReadStart) * 1000).count() << "\tms" << endl;

        return true;
    } else {
        return false;
    }
}



bool readInputFile()
{
    char buffer[2];
    int mirCount = 0;
    int sampleCount = 0;

    FILE *file;

    file = fopen(FILE_NAME, "r");
    if (file) {
        auto timeFileReadStart = chrono::system_clock::now();
        while (!feof(file)) {
			
            if (fgets(buffer, 2, file) == NULL){
				break;
			}

            if (*buffer != '\t' && *buffer != '\n') {

                if (mirCount >= 2) {

                    if (sampleCount != 1) {
                        (fileData[mirCount - 2][sampleCount == 0 ? sampleCount : sampleCount - 1]).append(buffer);
                        // (fileData[mirCount-1][sampleCount == 0 ? sampleCount : sampleCount]).append(buffer);

                    }
                }
            }


            
            if (*buffer == '\t') {
                sampleCount++;
            }

            if (*buffer == '\n') {
				// if(mirCount>99){
					// cout << "mirCount:" << mirCount << endl;
					// cout << "sampleCount:" << sampleCount << endl;
					// cout << fileData[mirCount - 2][sampleCount - 1];
				// }
					
				
                mirCount++;
                sampleCount = 0;
            }

        }

        // MIRS -= 2;
        // SAMPLES -= 1;
        fclose(file);
        auto timeFileReadEnd = chrono::system_clock::now();
        metaFile << "time\tfileread\t" << ((chrono::duration<double>)(timeFileReadEnd - timeFileReadStart) * 1000).count() << "\tms" << endl;

        return true;
    } else {
        return false;
    }
}

void displayInputData() {
    for (int i = 0; i <= MIRS; i++) {
        cout << i << ": ";
        for (int j = 0; j <= SAMPLES; j++) {
            cout << fileData[i][j] << "\t";
        }
        cout << endl;
    }
}

bool fileExists(const std::string& filename)
{
    struct stat buf;

    return (stat(filename.c_str(), &buf) != -1);
}

void filterInputData()
{
    int i, j, u;
    int mirsRemoved = 0;
    bool correctMir;
    float tmpData;

    int sameSamples = 0;

    for (i = 1; i <= MIRS; i++) {
        correctMir = true;
        //cout << "filterInputData::c1 i: " << i <<endl;
        for (j = 1; j <= SAMPLES; j++) {
            if (fileData[i][j].empty()) {
                correctMir = false;
                break;
            }

            try {
                tmpData = stof(fileData[i][j]);

                if ((METHOD == 1 && tmpData < 0.0) || (METHOD == 2 && tmpData < 1.0)) {
                    correctMir = false;
                    //cout << "filterInputData::c2 i: " << i <<endl;
                    break;
                }
            }
            catch (const std::invalid_argument& ia) {
                correctMir = false;
                //cout << "filterInputData::c3 i: " << i <<endl;
                break;
            }
        }

        if (correctMir) {
            for (u = 1; u <= MIRS; u++) {
                if (u != i) {
                    sameSamples = 0;
                    for (j = 1; j <= SAMPLES; j++) {
                        if (fileData[i][j] == fileData[u][j]) {
                            sameSamples++;
                        }
                    }

                    if (sameSamples == SAMPLES && i > u) {
                        correctMir = false;
                        break;
                    }
                }
            }
        }

        if (correctMir) {
            copy(&fileData[i][0], &fileData[i][SAMPLES+1], &fileData[i - mirsRemoved][0]);
        }
        else {
            mirsRemoved++;
        }
    }

    MIRS -= mirsRemoved;
    cout << "MIRS after filtering: " << MIRS << endl;
}

int end(bool error = false)
{
    remove(LOCK_FILE_NAME.c_str());
    metaFile.close();

#ifdef _WIN32
    cout << endl << "DONE";
    cin.ignore();
#endif

    return (error ? -1 : 0);
}

int interrupt()
{
    return end(true);
}


void deviceSetup()
{
    size_t free_t, total_t, total_m, used_m;

    checkCuda(cudaDeviceReset());
    checkCuda(cudaSetDevice(DEVICE_ID));
    checkCuda(cudaMemGetInfo(&free_t, &total_t));

    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, DEVICE_ID));
    cout << endl << "Device: " << prop.name << endl;
    cout << "Prop total global memory: " << prop.totalGlobalMem / 1048576.0 << endl;

    free_m = (free_t / 1024.0 / 1024.0);
    total_m = (total_t / 1024.0 / 1024.0);
    used_m = total_m - free_m;

    cout << "Total: " << total_m << endl
        << "Used: " << used_m << endl
        << "Free: " << free_m << endl;

    checkCuda(cudaDeviceSetLimit(cudaLimitMallocHeapSize, free_m));
}

void sighandler(int sig)
{
#ifdef _WIN32
    writeToErrorFile("Signal (" + to_string(sig) + ") caught...");
    cout << "Signal (" + to_string(sig) + ") caught..." << endl;
#endif
#ifdef __linux__
    writeToErrorFile("Signal \"" + (string)strsignal(sig) + "\" (" + to_string(sig) + ") caught...");
    cout << "Signal \"" + (string)strsignal(sig) + "\" (" + to_string(sig) + ") caught..." << endl;
#endif

    interrupt();

    exit(1);
}



int main(int argc, char** argv)
{
    /*cout << "Version 2.1.1" << endl;
    cout << "argc: " << argc << endl;
    cout << "argv[0]: " << argv[0] << endl;
    cout << "argv[1]: " << argv[1] << endl;
    cout << "argv[2]: " << argv[2] << endl;
    cout << "argv[3]: " << argv[3] << endl;
    cout << "argv[4]: " << argv[4] << endl;
    cout << "argv[5]: " << argv[5] << endl;
    cout << "argv[6]: " << argv[6] << endl;
    cout << "argv[7]: " << argv[7] << endl;
    cout << "argv[8]: " << argv[8] << endl;
    cout << "argv[9]: " << argv[9] << endl;
    cout << "argv[10]: " << argv[10] << endl;
    cout << "argv[11]: " << argv[11] << endl;
    cout << "argv[12]: " << argv[12] << endl;
    cout << "argv[13]: " << argv[13] << endl;
    cout << "argv[14]: " << argv[14] << endl;*/
    if (argc < 16 || (stoi(argv[1]) == 2 && argc < 17)) {
        cout << "How to run: " << endl
            << "  ./main [algo] [combLen] [fileName] [outFileName] [errFileName] [metaFileName] [mirs] [samples] [method] [groups*]" << endl
            << "    algo: 1 - GeNorm, 2 - NormFinder, 3 - BestKeeper" << endl
            << "    combLen: combination length" << endl
            << "    fileName: input file name (formatted correctly)" << endl
            << "    outFileName: output file name (does not have to exist)" << endl
            << "    errFileName: error file name (does not have to exist)" << endl
            << "    metaFileName: meta file name (does not have to exist)" << endl
            << "    mirs: number of mirs" << endl
            << "    samples: number of samples" << endl
            << "    method: 1 - qPCR (log), 2 - seq-uarray (expr)" << endl
            << "    groups (only for NormFinder): group division in form: 001112222" << endl;
        
        return interrupt();
    } else {
        //// START total
        auto timeTotalStart = chrono::system_clock::now();

        /*ofstream lockFile;
        lockFile.open(LOCK_FILE_NAME);
        lockFile << "locked" << endl;

        if (!fileExists(LOCK_FILE_NAME)) {
            cout << "Cannot create lock file!" << endl;
            writeToErrorFile("Cannot create lock file!");

            return interrupt();
        }*/

        //lockFile.close();

        //signal(SIGABRT, &sighandler);
        //signal(SIGTERM, &sighandler);
        //signal(SIGINT , &sighandler);
        //signal(SIGSEGV, &sighandler);
	int i = 0;

        ALGORITHM          = stoi(argv[1]);
        COMBINATION_LENGTH = stoi(argv[2]);
        COMBINATION_NUMBER = stoi(argv[3]);
        FILE_NAME          = argv[4];
        OUTPUT_FILE_NAME   = argv[5];
        ERROR_FILE_NAME    = argv[6];
        META_FILE_NAME     = argv[7];
        MIRS               = stoi(argv[8]);
	SAMPLES            = stoi(argv[9]);
        METHOD             = stoi(argv[10]);
        WEIGHT_FILE_NAME   = argv[11];
        COMBS_FILE_NAME    = argv[12];
        GEOMETRIC          = stoi(argv[13]);
        NUMOFUNSTABLEMIRS  = stoi(argv[14]);
	UNSTABLEMIRS_FILE_NAME  = argv[15];

        metaFile.open(META_FILE_NAME);
        
        if (ALGORITHM != 1 && ALGORITHM != 2 && ALGORITHM != 3) {
            writeToErrorFile("Incorrect algorithm number!");
            cout << "ERROR: Incorrect algorithm number!" << endl;

            return interrupt();
        }

        if (ALGORITHM == 2) {
            string groupsString = argv[16];

            if (groupsString.length() != SAMPLES) {
                writeToErrorFile("Incorrect groups length!");
                cout << "ERROR: Incorrect groups length!" << endl;

                return interrupt();
            }

            groups = (int*)malloc(SAMPLES * sizeof(int));

            for (i = 0; i < groupsString.length(); i++) {
                groups[i] = (int)groupsString[i] - 48;
                // groups[i] = (int)groupsString[i];
            }

            cout << "groups: " << endl;
            for (i = 0; i < SAMPLES; i++) {
                cout << groups[i] << " ";
            }
            cout << endl;
        }

        if (COMBINATION_LENGTH < 1 || COMBINATION_LENGTH > MIRS) {
            writeToErrorFile("Incorrect combination length!");
            cout << "ERROR: Incorrect combination length!" << endl;

            return interrupt();
        }

        if (METHOD != 1 && METHOD != 2) {
            writeToErrorFile("Incorrect method number!");
            cout << "ERROR: Incorrect method number!" << endl;
            
            return interrupt();
        }

        cout << "Algorithm: "          << (ALGORITHM == 1 ? "GeNorm" : ALGORITHM == 2 ? "NormFinder" : "BestKeeper") 
             << " (" << ALGORITHM << ")" << endl;
        cout << "Combination Length: " << COMBINATION_LENGTH << endl;
        cout << "File Name: "          << FILE_NAME << endl;
        cout << "Output File Name: "   << OUTPUT_FILE_NAME << endl;
        cout << "Error File Name: "    << ERROR_FILE_NAME << endl;
        cout << "Mir Number: "         << MIRS << endl;
        cout << "Sample Number: "      << SAMPLES << endl;
        cout << "Method: "             << (METHOD == 1 ? "qPCR (log)" : "seq-uarray (expr)") << endl;
        cout << "Weight File Name: "   << WEIGHT_FILE_NAME << endl;
        cout << "Combinations File Name: "   << COMBS_FILE_NAME << endl;
        cout << "GEOMETRIC: "          << (GEOMETRIC == 0 ? "No" : "Yes") << endl;
        cout << "NUMOFUNSTABLEMIRS: "  << NUMOFUNSTABLEMIRS << endl;
        cout << "Unstable miRs File Name: "   << UNSTABLEMIRS_FILE_NAME << endl;
	    
        fileData = new string*[MIRS + 1]; //(string**)malloc((MIRS + 1) * sizeof(string*));
        for (i = 0; i <= MIRS; i++) {
            fileData[i] =   new string[SAMPLES + 1];//(string*)calloc((SAMPLES + 1), sizeof(string));
        }

	UnstablemiRsHostFlat =	(int*) malloc(NUMOFUNSTABLEMIRS * sizeof(int));
		
        if (!readInputFile()) {
            writeToErrorFile("Input file does not exist!");
            cout << "ERROR: Input file does not exist!" << endl;
            
            return interrupt();
        }
        else {
            cout << "+ File read" << endl;
        }

        cout << "before filtering MIRS: " << MIRS << ", SAMPLES: " << SAMPLES << endl;
        // displayInputData();
        filterInputData();
        // displayInputData();
        
        cout << " after filtering MIRS: " << MIRS << ", SAMPLES: " << SAMPLES << endl;
        //return end();

        if (MIRS < 2) {
            writeToErrorFile("Too few MiRs!");
            cout << "ERROR: Too few MiRs!" << endl;

            return interrupt();
        }
		
		
        //// START Combination determination
        auto timeCombDetStart = chrono::system_clock::now();

        differences = (unsigned long long int*)malloc(COMBINATION_LENGTH * sizeof(unsigned long long int));
		
        //calculateCombinationNumber(); the COMBINATION_LENGTH is obtained in command
        cout << "Combination number: " << COMBINATION_NUMBER << endl;
		
        // determine weights
        WeightofMeanHostFlat = new float[COMBINATION_NUMBER*COMBINATION_LENGTH];
        if (!ReadWeightFile()) {
            writeToErrorFile("Weight file does not exist!");
            cout << "ERROR: Weight file does not exist!" << endl;
            
            return interrupt();
        }
        else {
            cout << "+ Weight File read" << endl;
        }

        // determine Combination Indices
        mirCombinationsFlat =  new int[COMBINATION_NUMBER * COMBINATION_LENGTH];
        //mirCombinationsFlat = (unsigned int*)malloc(COMBINATION_NUMBER * COMBINATION_LENGTH * sizeof(unsigned int));

        if (!ReadCombsFile()) {
            writeToErrorFile("Combination file does not exist!");
            cout << "ERROR: Combination file does not exist!" << endl;
            
            return interrupt();
        }
        else {
            cout << "+ Combination File read" << endl;
        }
		

        // determine weights
        UnstablemiRsHostFlat = new int[NUMOFUNSTABLEMIRS];
        if (!ReadUnstablemiRsFile()) {
            writeToErrorFile("UnstablemiRs file does not exist!");
            cout << "ERROR: UnstablemiRs file does not exist!" << endl;

            return interrupt();
        }
        else {
            cout << "+ UnstablemiRs File read" << endl;
        }
        auto timeCombDetEnd = chrono::system_clock::now();
        metaFile << "time\tcombdet\t" << ((chrono::duration<double>)(timeCombDetEnd - timeCombDetStart) * 1000).count() << "\tms" << endl;
        //// END Combination determination

        //// START device setup
        auto timeSetupStart = chrono::system_clock::now();
        deviceSetup();
        auto timeSetupEnd = chrono::system_clock::now();
        metaFile << "time\tsetup\t" << ((chrono::duration<double>)(timeSetupEnd - timeSetupStart) * 1000).count() << "\tms" << endl;
        //// END device setup

        //// START algorithm
        auto timeAlgorithmStart = chrono::system_clock::now();
        if (ALGORITHM == 1) {
            mainGeNorm();
        }
        else if (ALGORITHM == 2) {
            mainNormFinder();
        }
        else if (ALGORITHM == 3) {
            mainBestKeeper();
        }
        auto timeAlgorithmEnd = chrono::system_clock::now();
        metaFile << "time\talgorithm\t" << ((chrono::duration<double>)(timeAlgorithmEnd - timeAlgorithmStart) * 1000).count() << "\tms" << endl;
        //// END algorithm

        checkCuda(cudaDeviceReset());

        free(differences);
        /*for (i = 0; i <= MIRS; i++) {
            free(fileData[i]);
        }
        free(fileData);*/
		for (i = 0; i <= MIRS; i++) {
            delete[] fileData[i];
        }
        delete[] fileData;
        free(mirCombinationsFlat);
        free(indeces);

        auto timeTotalEnd = chrono::system_clock::now();
        metaFile << "time\ttotal\t" << ((chrono::duration<double>)(timeTotalEnd - timeTotalStart) * 1000).count() << "\tms" << endl;
        //// END total
    }

    return end();
}

