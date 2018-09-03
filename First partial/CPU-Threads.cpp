//Adrian Biller A01018940 matrix multiplication using cpu and threads



#include <omp.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>




using namespace std;

void printArray(int ** array, int size){
  for(int i = 0; i < size; i++){
    for(int j = 0; j < size; j++){
      cout<<" "<<array[i][j];
    }
    cout<<endl;
  }
  cout<<endl;
}



int** generateData(int size){
  int** array = new int*[size];
  for(int i = 0; i < size; i++){
    array[i] = new int [size];
    for(int j = 0; j < size; j++){
      array[i][j] = rand() % 10;
    }
  }
  // printArray(array, size);
  return array;
}



//
// int * multiplyMatrices(int** matrixA, int ** matrixB, int size){
//   int * multiplication  = new int[size];
//   for(int i = 0; i < size; i++){
//     multiplication[i] = new int[size];
//     for(int j = 0; j < size; j++){
//       for(int k = 0; k < size; k++){
//         multiplication[i][j] += matrixA[i][k] * matrixB[k][j];
//       }
//     }
//   }
//   return multiplication;
// }

int * multiplyMatrices(int** matrixA, int ** matrixB, int size, int threadID){
	int * multiplication = new int[size];
	for(int j = 0; j < size; j++){
		for(int k = 0; k < size; k++){
			multiplication[j] += matrixA[threadID][k] * matrixB[k][j];
		}
	}
	return multiplication;
}


void allThreads (){
	int threadID, totalThreads;


	srand(time(NULL));

	cout<< "1 for multiplication of 1000 x 1000 matrix"<<endl;
	cout<< "2 for multiplication of 2000 x 2000 matrix"<<endl;
	cout<< "3 for multiplication of 4000 x 4000 matrix"<<endl;
	int size = 0;
	cin>>size;
	if(size == 1)
		size = 1000;
	else if(size == 2)
		size = 2000;
	else
		size = 4000;
	int ** array1  = generateData(size);
	int ** array2  = generateData(size);
	int * multiplication  = new int[size];
	clock_t begin = clock();
	//setting number of threads related to size of matrix
	omp_set_num_threads(size);

	/* Fork a team of threads giving them their own copies of variables */
	#pragma omp parallel private(threadID) //num_threads(4)
	{
		threadID = omp_get_thread_num();
		// cout << "Hello world from thread: " << (int)threadID << endl;
		multiplication[(int)threadID] = multiplyMatrices(array1, array2, size, (int)threadID);
		// if (threadID == 0)
		// {
		// 	cout << "Master thread" << endl;
		// 	totalThreads = omp_get_num_threads();
		// 	cout << "Total threads: " << totalThreads << endl;
		// }

	}
	clock_t end = clock();

	printArray(multiplication, size);
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout<<"Elapsed time: "<<elapsed_secs<<" seconds"<<endl;
}




int main()
{
	// omp_set_num_threads(4);
	allThreads();
	return 0;
}
