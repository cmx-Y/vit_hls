#include "vit.h"
#include <iostream>
#include <assert.h>
#include <math.h>

str::str(){
	_size = 0;
}

str::str(const char *s){
	int cnt = 0;
	while(s[cnt] != '\0'){
		_str[cnt] = s[cnt];
		cnt++;
	}
	_size = cnt;
}

int str::size(){
	return _size;
}

char *str::getptr(){
	return _str;
}

void str::print(){
	for(int i = 0; i < _size; i++){
		std::cout << _str[i];
	}
}

void print_mat(str name, data_t* x, int row, int col){
	for(int i = 0; i < row; i++)
		for(int j = 0; j < col; j++){
			name.print();
			std::cout << "[" << i << "][" << j <<"]: " << x[i*col + j] << std::endl;
		}
}

void LayerNorm::ln(data_t* X_addr, int X_row, int X_col){
	double epsilon = 1e-8;
	for(int i = 0; i < X_row; i++){
		double sum = 0;
		double mean = 0;
		double var = 0;

		for(int j = 0; j < X_col; j++){
			sum += X_addr[i*X_col + j];
		}
		mean = sum / X_col;

		for(int j = 0; j < X_col; j++){
			var += (X_addr[i*X_col + j] - mean) * (X_addr[i*X_col + j] - mean);
		}
		var /= X_col;

		for(int j = 0; j < X_col; j++){
			X_addr[i*X_col + j] = (X_addr[i*X_col + j] - mean) / sqrt(var + epsilon);
		}
	}

	//output[i] = gamma_[i] * output[i] + beta_[i];
}

data_t gelu(data_t x){
	return 0.5 * x * (1 + tanh(sqrt(2/PI) * (x + 0.044715*x*x*x)));
}

void Gelu(data_t* X_addr, int size){
	for(int i = 0; i < size; i++){
		*(X_addr) = gelu(*(X_addr));
		X_addr++;
	}
}

void softmax(data_t* X_addr, int X_row, int X_col){
	for(int i = 0; i < X_row; i++){
		data_t esum = 0;
		for(int k = 0; k < X_col; k++){
			esum += exp(X_addr[i*X_col + k]);
		}

		for(int j = 0; j < X_col; j++){
			X_addr[i*X_col + j] = exp(X_addr[i*X_col + j]) / esum;
		}
	}

}

void matmul(data_t* A_addr, int A_row, int A_col, data_t* B_addr, int B_row, int B_col,data_t* C_addr){
	//assert(A_col == B_row);
	int C_row = A_row;
	int C_col = B_col;

	for(int i = 0; i < C_row; i++)
		for(int j = 0; j < C_col; j++){
			// C[i][j] = 0;
			C_addr[i*C_col + j] = 0;
			for(int k = 0; k < A_col; k++){
				// C[i][j] += A[i][k] * B[k][j];
				C_addr[i*C_col + j] += A_addr[i*A_col + k] * B_addr[k*B_col + j];
			}
		}
}

void matadd(data_t* A_addr,data_t* B_addr,data_t* C_addr, int row, int col){
	for(int i = 0; i < row; i++)
		for(int j = 0; j < col; j++){
			//C[i][j] = 0;
			C_addr[i*col + j] = 0;
			//C[i][j] = A[i][j] + B[i][j];
			C_addr[i*col + j] = A_addr[i*col + j] + B_addr[i*col + j];
		}
}

void linear(data_t* X_addr, data_t* W_addr, data_t* B_addr, data_t* XW_addr, data_t* Rs_addr,int sample_num, int in_features, int out_features){
	matmul(X_addr, sample_num, in_features, W_addr, in_features, out_features, XW_addr);
	matadd(XW_addr, B_addr, Rs_addr, sample_num, out_features);
}

void Mlp::forward(data_t* input_addr, data_t* output_addr){
	data_t XW0[MLP_NUM][MLP_HIDDEN_DIM];
	data_t XW1[MLP_NUM][MLP_OUT_FEATURES];
	data_t linear_rs0[MLP_NUM][MLP_HIDDEN_DIM];


	linear(input_addr, &(this->_W0[0][0]), &(this->_B0[0][0]), &XW0[0][0], &linear_rs0[0][0],
		   MLP_NUM, MLP_IN_FEATURES, MLP_HIDDEN_DIM);

	Gelu(&linear_rs0[0][0], MLP_NUM * MLP_OUT_FEATURES);

	linear(&linear_rs0[0][0], &(this->_W1[0][0]), &(this->_B1[0][0]), &XW1[0][0], output_addr,
		   MLP_NUM, MLP_HIDDEN_DIM, MLP_OUT_FEATURES);
}

void mlp_test(){
	data_t X[2][4] = {{1, 2, 3, 4}, {4, 5, 9, 7}};
	data_t W0[4][3] = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}};
	data_t B0[2][3] = {{3, 2, 1}, {3, 2, 1}};
	data_t W1[3][2] = {{1, 1}, {2, 2}, {3, 3}};
	data_t B1[2][2] = {{3, 2}, {3, 2}};
	data_t Rs[2][2];
	data_t Rs_std[2][2] = {{193, 192}, {427, 426}};

	Mlp mlp_block = Mlp("mlp_block001");
	mlp_block.setW0(&W0[0][0], 4, 3);
	mlp_block.setB0(&B0[0][0], 2, 3);
	mlp_block.setW1(&W1[0][0], 3, 2);
	mlp_block.setB1(&B1[0][0], 2, 2);
	mlp_block.forward(&X[0][0], &Rs[0][0]);

	for(int i = 0; i < 2; i++)
		for(int j = 0; j < 2; j++){
			assert(Rs[i][j] == Rs_std[i][j]);
		}
}

void Msa::forward(data_t* input_addr, data_t* output_addr){
	data_t Q[MSA_NUM][MSA_W_DIM];
	data_t K[MSA_NUM][MSA_W_DIM];
	data_t V[MSA_NUM][MSA_W_DIM];
	data_t KT[MSA_W_DIM][MSA_NUM];
	data_t QKT[MSA_NUM][MSA_NUM];
	data_t Attn[MSA_NUM][MSA_W_DIM];

	matmul(input_addr, MSA_NUM, MSA_IN_FEATURES, &(this->_Wq[0][0]), MSA_IN_FEATURES, MSA_W_DIM,
		   &Q[0][0]);
	matmul(input_addr, MSA_NUM, MSA_IN_FEATURES, &(this->_Wk[0][0]), MSA_IN_FEATURES, MSA_W_DIM,
		   &K[0][0]);
	matmul(input_addr, MSA_NUM, MSA_IN_FEATURES, &(this->_Wv[0][0]), MSA_IN_FEATURES, MSA_W_DIM,
		   &V[0][0]);

	for(int i = 0; i < MSA_NUM; i++)
		for(int j = 0; j < MSA_W_DIM; j++){
			KT[j][i] = K[i][j];
		}

	matmul(&Q[0][0], MSA_NUM, MSA_W_DIM, &KT[0][0], MSA_W_DIM, MSA_NUM, &QKT[0][0]);

	for(int i = 0; i < MSA_NUM; i++)
		for(int j = 0; j < MSA_NUM; j++){
			QKT[i][j] = QKT[i][j] / this->_scale;
		}

	print_mat("QKT", &QKT[0][0], 2, 2);
	softmax(&QKT[0][0], MSA_NUM, MSA_NUM);
	print_mat("softmax(QKT)", &QKT[0][0], 2, 2);
	print_mat("V", &V[0][0], 2, 3);

	matmul(&QKT[0][0], MSA_NUM, MSA_NUM, &V[0][0], MSA_NUM, MSA_W_DIM, &Attn[0][0]);
	print_mat("Attn", &Attn[0][0], MSA_NUM, MSA_W_DIM);

	matmul(&Attn[0][0], MSA_NUM, MSA_W_DIM, &(this->_Wproj[0][0]), MSA_W_DIM, MSA_OUT_FEATURES,
		   output_addr);

}

void msa_test(){
	data_t X[2][4] = {{1, 2, 33, 4}, {4, 5, 9, 7}};
	data_t Wq[4][3] = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}};
	data_t Wproj[3][2] = {{1, 1}, {2, 2}, {3, 3}};

	data_t Rs[2][2];

	Msa msa_block = Msa("msa_block001");
	msa_block.setWq(&Wq[0][0], 4, 3);
	msa_block.setWk(&Wq[0][0], 4, 3);
	msa_block.setWv(&Wq[0][0], 4, 3);
	msa_block.setWproj(&Wproj[0][0], 3, 2);
	msa_block.setScale(10000);
	msa_block.forward(&X[0][0], &Rs[0][0]);
	print_mat("Rs", &Rs[0][0], 2, 2);
}

void TransBlock::forward(data_t* input_addr, data_t* output_addr){
	data_t X0[MSA_NUM][MSA_IN_FEATURES];
	for(int i = 0; i < MSA_NUM; i++)
		for(int j = 0; j < MSA_IN_FEATURES; j++){
			X0[i][j] = input_addr[i*MSA_IN_FEATURES + j];
		}

	LayerNorm ln0 = LayerNorm("LayerNorm0");
	ln0.ln(input_addr, MSA_NUM, MSA_IN_FEATURES);

	data_t X2[MSA_NUM][MSA_OUT_FEATURES];
	Msa msa_block = Msa("msa_block");
	msa_block.forward(input_addr, &X2[0][0]);

	data_t X3[MSA_NUM][MSA_OUT_FEATURES];
	matadd(&X0[0][0], &X2[0][0], &X3[0][0], MSA_NUM, MSA_OUT_FEATURES);

	data_t X4[MSA_NUM][MSA_OUT_FEATURES];
	for(int i = 0; i < MSA_NUM; i++)
		for(int j = 0; j < MSA_OUT_FEATURES; j++){
			X4[i][j] = X3[i][j];
		}
	LayerNorm ln1 = LayerNorm("LayerNorm1");
	ln1.ln(&X4[0][0], MSA_NUM, MSA_OUT_FEATURES);


	data_t X5[MLP_NUM][MLP_OUT_FEATURES];
	Mlp	mlp_block = Mlp("mlp_block");
	mlp_block.forward(&X4[0][0], &X5[0][0]);

	matadd(&X3[0][0], &X5[0][0], output_addr, MLP_NUM, MLP_OUT_FEATURES);
}

void vit()
{
	mlp_test();
	msa_test();
}
