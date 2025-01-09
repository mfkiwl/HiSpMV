/*
Header file including necessary nvml headers.
*/

#ifndef INCLNVML
#define INCLNVML

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <nvml.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <time.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <chrono>
#include <thread>

void nvmlAPIRun(char* filename);
void nvmlAPIEnd();
void *powerPollingFunc(void *ptr, char* filename);
int getNVMLError(nvmlReturn_t resultToCheck);

#endif
