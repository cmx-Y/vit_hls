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
	for(int i = 0; i < row; i++){
		for(int j = 0; j < col; j++){
			name.print();
			std::cout << "[" << i << "][" << j <<"]: " << x[i*col + j] << "   ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void print_mat_3d(str name, data_t* x, int dim0, int dim1, int dim2){
	for(int i = 0; i < dim0; i++)
		for(int j = 0; j < dim1; j++){
			for(int k = 0; k < dim2; k++){
				name.print();
				std::cout << "[" << i << "][" << j <<"][" << k << "]: " <<
						x[i*dim1*dim2 + j*dim2 + k] << "   ";
			}
			std::cout << std::endl;
		}
	std::cout << std::endl;
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

void nhd2hnd(data_t* nhd_addr, data_t* hnd_addr){
	int nhd_axis2_stride = 1;
	int nhd_axis2_size = DIM_HEAD;
	int nhd_axis1_stride = nhd_axis2_size * nhd_axis2_stride;
	int nhd_axis1_size = HEADS;
	int nhd_axis0_stride = nhd_axis1_size * nhd_axis1_stride;
	int nhd_axis0_size = INPUT_NUM;

	int hnd_axis2_stride = nhd_axis2_stride;
	int hnd_axis2_size = nhd_axis2_size;
	int hnd_axis1_stride = nhd_axis0_stride;
	int hnd_axis1_size = nhd_axis0_size;
	int hnd_axis0_stride = nhd_axis1_stride;
	int hnd_axis0_size = nhd_axis1_size;

	for(int i = 0; i < HEADS; i++)
		for(int j = 0; j < INPUT_NUM; j++)
			for(int k = 0; k < DIM_HEAD; k++){
				hnd_addr[i*INPUT_NUM*DIM_HEAD + j*DIM_HEAD + k]
						 = nhd_addr[i*hnd_axis0_stride + j*hnd_axis1_stride + k];
			}
}

void hnd2nhd(data_t* hnd_addr, data_t* nhd_addr){
	int hnd_axis2_stride = 1;
	int hnd_axis2_size = DIM_HEAD;
	int hnd_axis1_stride = hnd_axis2_size * hnd_axis2_stride;
	int hnd_axis1_size = INPUT_NUM;
	int hnd_axis0_stride = hnd_axis1_size * hnd_axis1_stride;
	int hnd_axis0_size = HEADS;

	int nhd_axis2_stride = hnd_axis2_stride;
	int nhd_axis2_size = hnd_axis2_size;
	int nhd_axis1_stride = hnd_axis0_stride;
	int nhd_axis1_size = hnd_axis0_size;
	int nhd_axis0_stride = hnd_axis1_stride;
	int nhd_axis0_size = hnd_axis1_size;

	for(int i = 0; i < INPUT_NUM; i++)
		for(int j = 0; j < HEADS; j++)
			for(int k = 0; k < DIM_HEAD; k++){
				nhd_addr[i*HEADS*DIM_HEAD + j*DIM_HEAD + k]
						= hnd_addr[i*nhd_axis0_stride + j*nhd_axis1_stride + k];
			}
}

void increDim(data_t* input_addr, data_t* output_addr){
	//n (h d) -> n h d
	int output_axis2_stride = 1;
	int output_axis1_stride = DIM_HEAD;
	int output_axis0_stride = HEADS * output_axis1_stride;
	for(int i = 0; i < INPUT_NUM; i++)
		for(int j = 0; j < HEADS; j++)
			for(int k = 0; k < DIM_HEAD; k++){
				output_addr[i*output_axis0_stride + j*output_axis1_stride + k]
							= input_addr[i*output_axis0_stride + j*output_axis1_stride + k];
			}
}

void decreDim(data_t* input_addr, data_t* output_addr){
	//n h d -> n (h d)

	for(int i = 0; i < INPUT_NUM; i++)
		for(int j = 0; j < HEADS; j++)
			for(int k = 0; k < DIM_HEAD; k++){
				output_addr[i*HEADS*DIM_HEAD + j*DIM_HEAD + k] = input_addr[i*HEADS*DIM_HEAD + j*DIM_HEAD + k];
			}
}

void rearrange_n_hd2hnd(data_t* input_addr, data_t* output_addr){
	data_t mid[INPUT_NUM][HEADS][DIM_HEAD];
	increDim(input_addr, &mid[0][0][0]);
	nhd2hnd(&mid[0][0][0], output_addr);
}

void rearrange_hnd2n_hd(data_t* input_addr, data_t* output_addr){
	data_t mid[INPUT_NUM][HEADS][DIM_HEAD];
	hnd2nhd(input_addr, &mid[0][0][0]);
	decreDim(&mid[0][0][0], output_addr);
}

void Q_KT_matmul(data_t* Q_addr, data_t* K_addr, data_t* QKT_addr){
	//Q:HEADS, INPUT_NUM, DIM_HEAD; K:HEADS, INPUT_NUM, DIM_HEAD
	//QKT:HEADS, INPUT_NUM, INPUT_NUM
	data_t KT[HEADS][DIM_HEAD][INPUT_NUM];
	data_t QKT[HEADS][INPUT_NUM][INPUT_NUM];
	for(int i = 0; i < HEADS; i++)
		for(int j = 0; j < DIM_HEAD; j++)
			for(int k = 0; k < INPUT_NUM; k++){
				KT[i][j][k] = K_addr[i*INPUT_NUM*DIM_HEAD + k*DIM_HEAD + j];
			}

	for(int i = 0; i < HEADS; i++){
		matmul(&Q_addr[i*INPUT_NUM*DIM_HEAD], INPUT_NUM, DIM_HEAD,
			   &KT[i][0][0], DIM_HEAD, INPUT_NUM, &QKT_addr[i*INPUT_NUM*INPUT_NUM]);
	}
}

void QKT_V_matmul(data_t* QKT_addr, data_t* V_addr, data_t* Attn_addr){
	//QKT:HEADS, INPUT_NUM, INPUT_NUM
	//V:HEADS, INPUT_NUM, DIM_HEAD
	//Attn:HEADS, INPUT_NUM, DIM_HEAD
	for(int i = 0; i < HEADS; i++){
		matmul(&QKT_addr[i*INPUT_NUM*INPUT_NUM], INPUT_NUM, INPUT_NUM,
			   &V_addr[i*INPUT_NUM*DIM_HEAD], INPUT_NUM, DIM_HEAD, &Attn_addr[i*INPUT_NUM*DIM_HEAD]);
	}
}

void Msa::forward(data_t* input_addr, data_t* output_addr){
	data_t Q[MSA_NUM][MSA_W_DIM];
	data_t K[MSA_NUM][MSA_W_DIM];
	data_t V[MSA_NUM][MSA_W_DIM];
	data_t Qhnd[HEADS][INPUT_NUM][DIM_HEAD];
	data_t Khnd[HEADS][INPUT_NUM][DIM_HEAD];
	data_t Vhnd[HEADS][INPUT_NUM][DIM_HEAD];
	data_t QKThnn[HEADS][INPUT_NUM][INPUT_NUM];
	data_t Attn_hnd[HEADS][INPUT_NUM][DIM_HEAD];
	data_t Attn_nhd[INPUT_NUM][HEADS][DIM_HEAD];
	data_t Attn[MSA_NUM][MSA_W_DIM];

	matmul(input_addr, MSA_NUM, MSA_IN_FEATURES, &(this->_Wq[0][0]), MSA_IN_FEATURES, MSA_W_DIM,
		   &Q[0][0]);
	matmul(input_addr, MSA_NUM, MSA_IN_FEATURES, &(this->_Wk[0][0]), MSA_IN_FEATURES, MSA_W_DIM,
		   &K[0][0]);
	matmul(input_addr, MSA_NUM, MSA_IN_FEATURES, &(this->_Wv[0][0]), MSA_IN_FEATURES, MSA_W_DIM,
		   &V[0][0]);

	rearrange_n_hd2hnd(&Q[0][0], &Qhnd[0][0][0]);
	rearrange_n_hd2hnd(&K[0][0], &Khnd[0][0][0]);
	rearrange_n_hd2hnd(&V[0][0], &Vhnd[0][0][0]);

	Q_KT_matmul(&Qhnd[0][0][0], &Khnd[0][0][0], &QKThnn[0][0][0]);
	for(int i = 0; i < HEADS; i++)
		for(int j = 0; j < INPUT_NUM; j++)
			for(int k = 0; k < INPUT_NUM; k++){
				QKThnn[i][j][k] = QKThnn[i][j][k] / this->_scale;
			}
	for(int i = 0; i < HEADS; i++){
		softmax(&QKThnn[i][0][0], INPUT_NUM, INPUT_NUM);
	}
	QKT_V_matmul(&QKThnn[0][0][0], &Vhnd[0][0][0], &Attn_hnd[0][0][0]);
	hnd2nhd(&Attn_hnd[0][0][0], &Attn_nhd[0][0][0]);
	rearrange_hnd2n_hd(&Attn_hnd[0][0][0], &Attn[0][0]);
	matmul(&Attn[0][0], MSA_NUM, MSA_W_DIM, &(this->_Wproj[0][0]), MSA_W_DIM, MSA_OUT_FEATURES,
		   output_addr);
}

void msa_test(){
	data_t X[4][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
	data_t Wq[3][6] = {{1, 1, 1, 2, 2, 2}, {3, 3, 3, 4, 4, 4}, {5, 5, 5, 6, 6, 6}};
	data_t Wproj[6][4] = {{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}, {5, 5, 5, 5}, {6, 6, 6, 6}};

	data_t Rs[4][4];

	Msa msa_block = Msa("msa_block001");
	msa_block.setWq(&Wq[0][0], 3, 6);
	msa_block.setWk(&Wq[0][0], 3, 6);
	msa_block.setWv(&Wq[0][0], 3, 6);
	msa_block.setWproj(&Wproj[0][0], 6, 4);
	msa_block.setScale(10000);
	msa_block.forward(&X[0][0], &Rs[0][0]);
	print_mat("Rs", &Rs[0][0], 4, 4);
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

void rearrange_test(){
	data_t input[4][6] = {{0, 1, 2, 3, 4, 5}, {8, 9, 10, 11, 12, 13},{80, 90, 100, 110, 120, 130},{81, 91, 101, 111, 121, 131}};
	data_t mid[4][2][3];
	data_t output[2][4][3];

	increDim(&input[0][0], &mid[0][0][0]);
	for(int i = 0; i < 4; i++)
		for(int j = 0; j < 2; j++)
			for(int k = 0; k < 3; k++){
				std::cout << "mid[" << i << "][" << j <<"][" << k << "]: " << mid[i][j][k] << std::endl;
			}

	nhd2hnd(&mid[0][0][0], &output[0][0][0]);

	for(int i = 0; i < 2; i++)
		for(int j = 0; j < 4; j++)
			for(int k = 0; k < 3; k++){
				std::cout << "output[" << i << "][" << j <<"][" << k << "]: " << output[i][j][k] << std::endl;
			}
}

void Q_KT_test(){
	data_t Q_ini[4][6] = {{0, 1, 2, 3, 4, 5}, {8, 9, 10, 11, 12, 13},{80, 90, 100, 110, 120, 130},{81, 91, 101, 111, 121, 131}};
	data_t Q_nhd[4][2][3];
	data_t Q_hnd[2][4][3];

	data_t K_ini[4][6] = {{0, 1, 2, 3, 4, 5}, {8, 9, 10, 11, 12, 13},{80, 90, 100, 110, 120, 130},{81, 91, 101, 111, 121, 131}};
	data_t K_nhd[4][2][3];
	data_t K_hnd[2][4][3];
	data_t QKT[2][4][4];
	increDim(&Q_ini[0][0], &Q_nhd[0][0][0]);
	nhd2hnd(&Q_nhd[0][0][0], &Q_hnd[0][0][0]);

	increDim(&K_ini[0][0], &K_nhd[0][0][0]);
	nhd2hnd(&K_nhd[0][0][0], &K_hnd[0][0][0]);
	Q_KT_matmul(&Q_hnd[0][0][0], &K_hnd[0][0][0], &QKT[0][0][0]);

	print_mat_3d("QKT", &QKT[0][0][0], 2, 4, 4);
}

void vit()
{
	//mlp_test();
	msa_test();
	//rearrange_test();
	//Q_KT_test();
}
