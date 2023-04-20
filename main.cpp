#include "vit.h"
#include <iostream>
 
int main()
{
	//data_t A[2][3] = {{1, 1, 1}, {2, 2, 2}};
	//data_t B[3][2] = {{1, 1}, {2, 2}, {3,3}};

	//data_t input[INPUT_NUM][D_MODEL] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
	data_t input[INPUT_NUM][D_MODEL] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
	data_t output[INPUT_NUM][D_MODEL];
 
	vit(&input[0][0], &output[0][0]);
 
	return 0;
}
