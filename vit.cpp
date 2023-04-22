#include "vit.h"
#include <iostream>
#include <assert.h>
#include <cmath>

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
#pragma HLS ARRAY_PARTITION variable=X_addr dim=1 complete
	data_t epsilon = 1e-8;
	data_t X[INPUT_NUM][D_MODEL];
#pragma HLS ARRAY_PARTITION variable=X dim=1 complete
	data_t sum[INPUT_NUM];
	data_t mean[INPUT_NUM];
	data_t var[INPUT_NUM];

	ln_sum_loop_i:
	for(int i = 0; i < X_row; i++){
#pragma HLS PIPELINE
		sum[i] = 0;
		ln_sum_loop_j:
		for(int j = 0; j < X_col; j++){
#pragma HLS UNROLL
			sum[i] += X_addr[i*X_col + j];
		}
		mean[i] = sum[i] / X_col;
	}

	ln_var_loop_i:
	for(int i = 0; i < X_row; i++){
#pragma HLS PIPELINE
		var[i] = 0;
		ln_var_loop_j:
		for(int j = 0; j < X_col; j++){
#pragma HLS UNROLL
			var[i] += (X_addr[i*X_col + j] - mean[i]) * (X_addr[i*X_col + j] - mean[i]);
		}
		var[i] /= X_col;
	}

	ln_rs_loop_i:
	for(int i = 0; i < X_row; i++){
#pragma HLS PIPELINE
		ln_rs_loop_j:
		for(int j = 0; j < X_col; j++){
#pragma HLS UNROLL
			X[i][j] = (X_addr[i*X_col + j] - mean[i]) / sqrt(var[i] + epsilon);
		}
	}

	ln_2X_addr_out:
	for(int i = 0; i < X_row; i++)
#pragma HLS PIPELINE
		ln_2X_addr_inner:
		for(int j = 0; j < X_col; j++){
#pragma HLS UNROLL
			X_addr[i*X_col + j] = X[i][j];
		}

	//output[i] = gamma_[i] * output[i] + beta_[i];
}

data_t gelu(data_t x){
	data_t pi = 3.1415926;
	return (data_t)0.5 * x * ((data_t)1 + tanh(sqrt(2/pi) * (x + (data_t)0.044715*x*x*x)));
}

void Gelu(data_t* X_addr, int X_row, int X_col){
	data_t X[INPUT_NUM][MLP_HIDDEN_DIM];

	Gelu_X_loop_i:
	for (int i = 0; i < INPUT_NUM; i++)
		Gelu_X_loop_j:
		for (int j = 0; j < MLP_HIDDEN_DIM; j++){
			X[i][j] = X_addr[i*X_col + j];
		}

	Gelu_loop_i:
	for(int i = 0; i < X_row; i++){
#pragma HLS PIPELINE
		Gelu_loop_j:
		for(int j = 0; j < X_col; j++){
#pragma HLS UNROLL
			X[i][j] = gelu(X[i][j]);
	 	}
	}

	Gelu_Xaddr_loop_i:
	for (int i = 0; i < INPUT_NUM; i++)
		Gelu_Xaddr_loop_j:
		for (int j = 0; j < MLP_HIDDEN_DIM; j++){
			X_addr[i*X_col + j] = X[i][j];
		}
}

void softmax(data_t* X_addr, int X_row, int X_col){
	softmax_loop_i:
	for(int i = 0; i < X_row; i++){
		data_t esum = 0;
		softmax_loop_k:
		for(int k = 0; k < X_col; k++){
			esum += hls::exp(X_addr[i*X_col + k]);
		}

		softmax_loop_j:
		for(int j = 0; j < X_col; j++){
			X_addr[i*X_col + j] = hls::exp(X_addr[i*X_col + j]) / esum;
		}
	}
}

void matmul(data_t* A_addr, int A_row, int A_col, data_t* B_addr, int B_row, int B_col,data_t* C_addr){
	//assert(A_col == B_row);
	int C_row = A_row;
	int C_col = B_col;

	matmul_i_loop:
	for(int i = 0; i < C_row; i++)
		matmul_j_loop:
		for(int j = 0; j < C_col; j++){
			C_addr[i*C_col + j] = 0;
			matmul_k_loop:
			for(int k = 0; k < A_col; k++){
				C_addr[i*C_col + j] += A_addr[i*A_col + k] * B_addr[k*B_col + j];
			}
		}
}

void matadd(data_t* A_addr,data_t* B_addr,data_t* C_addr, int row, int col){
	matadd_loop_i:
	for(int i = 0; i < row; i++)
#pragma HLS PIPELINE
		matadd_loop_j:
		for(int j = 0; j < col; j++){
#pragma HLS UNROLL
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
	data_t XW0[INPUT_NUM][MLP_HIDDEN_DIM];
	data_t XW1[INPUT_NUM][D_MODEL];
	data_t linear_rs0[INPUT_NUM][MLP_HIDDEN_DIM];


	linear(input_addr, &(this->_W0[0][0]), &(this->_B0[0][0]), &XW0[0][0], &linear_rs0[0][0],
		   INPUT_NUM, D_MODEL, MLP_HIDDEN_DIM);

	Gelu(&linear_rs0[0][0], INPUT_NUM, MLP_HIDDEN_DIM);

	linear(&linear_rs0[0][0], &(this->_W1[0][0]), &(this->_B1[0][0]), &XW1[0][0], output_addr,
		   INPUT_NUM, MLP_HIDDEN_DIM, D_MODEL);
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

	nhd2hnd_loop_i:
	for(int i = 0; i < HEADS; i++)
#pragma HLS PIPELINE
		nhd2hnd_loop_j:
		for(int j = 0; j < INPUT_NUM; j++)
#pragma HLS PIPELINE
			nhd2hnd_loop_k:
			for(int k = 0; k < DIM_HEAD; k++){
#pragma HLS UNROLL
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

	hnd2nhd_loop_i:
	for(int i = 0; i < INPUT_NUM; i++)
#pragma HLS PIPELINE
		hnd2nhd_loop_j:
		for(int j = 0; j < HEADS; j++)
#pragma HLS PIPELINE
			hnd2nhd_loop_k:
			for(int k = 0; k < DIM_HEAD; k++){
#pragma HLS UNROLL
				nhd_addr[i*HEADS*DIM_HEAD + j*DIM_HEAD + k]
						= hnd_addr[i*nhd_axis0_stride + j*nhd_axis1_stride + k];
			}
}

void increDim(data_t* input_addr, data_t* output_addr){
	//n (h d) -> n h d
	int output_axis2_stride = 1;
	int output_axis1_stride = DIM_HEAD;
	int output_axis0_stride = HEADS * output_axis1_stride;
	increDim_loop_i:
	for(int i = 0; i < INPUT_NUM; i++)
#pragma HLS PIPELINE
		increDim_loop_j:
		for(int j = 0; j < HEADS; j++)
#pragma HLS PIPELINE
			increDim_loop_k:
			for(int k = 0; k < DIM_HEAD; k++){
#pragma HLS UNROLL
				output_addr[i*output_axis0_stride + j*output_axis1_stride + k]
							= input_addr[i*output_axis0_stride + j*output_axis1_stride + k];
			}
}

void decreDim(data_t* input_addr, data_t* output_addr){
	//n h d -> n (h d)

	decreDim_loop_i:
	for(int i = 0; i < INPUT_NUM; i++)
#pragma HLS PIPELINE
		decreDim_loop_j:
		for(int j = 0; j < HEADS; j++)
#pragma HLS PIPELINE
			decreDim_loop_k:
			for(int k = 0; k < DIM_HEAD; k++){
#pragma HLS UNROLL
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
	KT_loop_i:
	for(int i = 0; i < HEADS; i++)
#pragma HLS PIPELINE
		KT_loop_j:
		for(int j = 0; j < DIM_HEAD; j++)
#pragma HLS PIPELINE
			KT_loop_k:
			for(int k = 0; k < INPUT_NUM; k++){
#pragma HLS UNROLL
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
	data_t Q[INPUT_NUM][MSA_HIDDEN_DIM];
	data_t K[INPUT_NUM][MSA_HIDDEN_DIM];
	data_t V[INPUT_NUM][MSA_HIDDEN_DIM];
	data_t Qhnd[HEADS][INPUT_NUM][DIM_HEAD];
	data_t Khnd[HEADS][INPUT_NUM][DIM_HEAD];
	data_t Vhnd[HEADS][INPUT_NUM][DIM_HEAD];
	data_t QKThnn[HEADS][INPUT_NUM][INPUT_NUM];
	data_t Attn_hnd[HEADS][INPUT_NUM][DIM_HEAD];
	data_t Attn_nhd[INPUT_NUM][HEADS][DIM_HEAD];
	data_t Attn[INPUT_NUM][MSA_HIDDEN_DIM];

	matmul(input_addr, INPUT_NUM, D_MODEL, &(this->_Wq[0][0]), D_MODEL, MSA_HIDDEN_DIM,
		   &Q[0][0]);
	matmul(input_addr, INPUT_NUM, D_MODEL, &(this->_Wk[0][0]), D_MODEL, MSA_HIDDEN_DIM,
		   &K[0][0]);
	matmul(input_addr, INPUT_NUM, D_MODEL, &(this->_Wv[0][0]), D_MODEL, MSA_HIDDEN_DIM,
		   &V[0][0]);

	rearrange_n_hd2hnd(&Q[0][0], &Qhnd[0][0][0]);
	rearrange_n_hd2hnd(&K[0][0], &Khnd[0][0][0]);
	rearrange_n_hd2hnd(&V[0][0], &Vhnd[0][0][0]);

	Q_KT_matmul(&Qhnd[0][0][0], &Khnd[0][0][0], &QKThnn[0][0][0]);
	
	divscale_loop_i:
	for(int i = 0; i < HEADS; i++)
		divscale_loop_j:
		for(int j = 0; j < INPUT_NUM; j++)
			divscale_loop_k:
			for(int k = 0; k < INPUT_NUM; k++){
#pragma HLS UNROLL
				QKThnn[i][j][k] = QKThnn[i][j][k] / this->_scale;
			}
	for(int i = 0; i < HEADS; i++){
		softmax(&QKThnn[i][0][0], INPUT_NUM, INPUT_NUM);
	}
	QKT_V_matmul(&QKThnn[0][0][0], &Vhnd[0][0][0], &Attn_hnd[0][0][0]);
	hnd2nhd(&Attn_hnd[0][0][0], &Attn_nhd[0][0][0]);
	rearrange_hnd2n_hd(&Attn_hnd[0][0][0], &Attn[0][0]);
	matmul(&Attn[0][0], INPUT_NUM, MSA_HIDDEN_DIM, &(this->_Wproj[0][0]), MSA_HIDDEN_DIM, D_MODEL,
		   output_addr);
}

void mlp_test(){
	// INPUT_NUM 4
	// D_MODEL 3
	// MLP_HIDDEN_DIM 3
	data_t X[4][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
	data_t W0[3][3] = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}};
	data_t B0[4][3] = {{3, 2, 1}, {3, 2, 1}, {3, 2, 1}, {3, 2, 1}};
	data_t W1[3][3] = {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}};
	data_t B1[4][3] = {{3, 2, 3}, {3, 2, 3}, {3, 2, 3}, {3, 2, 3}};
	data_t Rs[4][3];
	data_t Rs_std[4][3] = {{98, 97, 98}, {206, 205, 206}, {314, 313, 314}, {422, 421, 422}};

	Mlp mlp_block = Mlp("mlp_block001");
	mlp_block.setW0(&W0[0][0], 3, 3);
	mlp_block.setB0(&B0[0][0], 4, 3);
	mlp_block.setW1(&W1[0][0], 3, 3);
	mlp_block.setB1(&B1[0][0], 4, 3);
	mlp_block.forward(&X[0][0], &Rs[0][0]);

	print_mat("Mlp_rs", &Rs[0][0], 4, 3);
	for(int i = 0; i < 4; i++)
		for(int j = 0; j < 3; j++){
			assert(Rs[i][j] == Rs_std[i][j]);
		}
}

void msa_test(){
	data_t X[4][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
	data_t Wq[3][6] = {{1, 1, 1, 2, 2, 2}, {3, 3, 3, 4, 4, 4}, {5, 5, 5, 6, 6, 6}};
	data_t Wproj[6][3] = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}, {5, 5, 5}, {6, 6, 6}};

	data_t Rs[4][3];

	Msa msa_block = Msa("msa_block001");
	msa_block.setWq(&Wq[0][0], 3, 6);
	msa_block.setWk(&Wq[0][0], 3, 6);
	msa_block.setWv(&Wq[0][0], 3, 6);
	msa_block.setWproj(&Wproj[0][0], 6, 3);
	msa_block.setScale(10000);
	msa_block.forward(&X[0][0], &Rs[0][0]);
	print_mat("Msa_rs", &Rs[0][0], 4, 3);
}

void TransBlock::init(){
	data_t Wq[D_MODEL][MSA_HIDDEN_DIM];
	for (int i = 0; i < D_MODEL; i++)
		for (int j = 0; j < MSA_HIDDEN_DIM; j++){
			Wq[i][j] = 1;
		}
	data_t Wproj[MSA_HIDDEN_DIM][D_MODEL];
	for (int i = 0; i < MSA_HIDDEN_DIM; i++)
		for (int j = 0; j < D_MODEL; j++){
			Wproj[i][j] = 1;
		}
	this->msa_block.setWq(&Wq[0][0], D_MODEL, MSA_HIDDEN_DIM);
	this->msa_block.setWk(&Wq[0][0], D_MODEL, MSA_HIDDEN_DIM);
	this->msa_block.setWv(&Wq[0][0], D_MODEL, MSA_HIDDEN_DIM);
	this->msa_block.setWproj(&Wproj[0][0], MSA_HIDDEN_DIM, D_MODEL);
	this->msa_block.setScale(10000);
	
	data_t W0[D_MODEL][MLP_HIDDEN_DIM];
	data_t B0[INPUT_NUM][MLP_HIDDEN_DIM];
	data_t W1[MLP_HIDDEN_DIM][D_MODEL];
	data_t B1[INPUT_NUM][D_MODEL];
	for (int i = 0; i < D_MODEL; i++)
		for (int j = 0; j < MLP_HIDDEN_DIM; j++){
			W0[i][j] = 1;
		}
	for (int i = 0; i < INPUT_NUM; i++)
		for (int j = 0; j < MLP_HIDDEN_DIM; j++){
			B0[i][j] = 1;
		}
	for (int i = 0; i < MLP_HIDDEN_DIM; i++)
		for (int j = 0; j < D_MODEL; j++){
			W1[i][j] = 1;
		}
	for (int i = 0; i < INPUT_NUM; i++)
		for (int j = 0; j < D_MODEL; j++){
			B1[i][j] = 1;
		}

	this->mlp_block.setW0(&W0[0][0], D_MODEL, MLP_HIDDEN_DIM);
	this->mlp_block.setB0(&B0[0][0], INPUT_NUM, MLP_HIDDEN_DIM);
	this->mlp_block.setW1(&W1[0][0], MLP_HIDDEN_DIM, D_MODEL);
	this->mlp_block.setB1(&B1[0][0], INPUT_NUM, D_MODEL);
}

void TransBlock::forward(data_t* input_addr, data_t* output_addr){
	data_t X0[INPUT_NUM][D_MODEL];
	
	trans_block_X0_loop_i:
	for(int i = 0; i < INPUT_NUM; i++)
#pragma HLS PIPELINE
		trans_block_X0_loop_j:
		for(int j = 0; j < D_MODEL; j++){
#pragma HLS UNROLL
			X0[i][j] = input_addr[i*D_MODEL + j];
		}

	
	this->ln0.ln(input_addr, INPUT_NUM, D_MODEL);

	data_t X2[INPUT_NUM][D_MODEL];
	this->msa_block.forward(input_addr, &X2[0][0]);

	data_t X3[INPUT_NUM][D_MODEL];
	matadd(&X0[0][0], &X2[0][0], &X3[0][0], INPUT_NUM, D_MODEL);

	data_t X4[INPUT_NUM][D_MODEL];
	
	trans_block_X32X4_loop_i:
	for(int i = 0; i < INPUT_NUM; i++)
#pragma HLS PIPELINE
		trans_block_X32X4_loop_j:
		for(int j = 0; j < D_MODEL; j++){
#pragma HLS UNROLL
			X4[i][j] = X3[i][j];
		}
	
	this->ln1.ln(&X4[0][0], INPUT_NUM, D_MODEL);


	data_t X5[INPUT_NUM][D_MODEL];
	this->mlp_block.forward(&X4[0][0], &X5[0][0]);

	matadd(&X3[0][0], &X5[0][0], output_addr, INPUT_NUM, D_MODEL);
}

void vit(data_t* input_addr, data_t* output_addr)
{
	//mlp_test();
	//msa_test();
	TransBlock trans_block;
	trans_block.init();
	trans_block.forward(input_addr, output_addr);
	print_mat("out", output_addr, INPUT_NUM, D_MODEL);
}
