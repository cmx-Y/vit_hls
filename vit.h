#ifndef __VIT__
#define __VIT__

#include "ap_fixed.h"
#include "hls_math.h"

#define PI 3.1415926

#define INPUT_NUM 4
#define D_MODEL 24
#define MLP_HIDDEN_DIM 12
#define MSA_HIDDEN_DIM 24
#define HEADS 4
#define DIM_HEAD 6

//MSA_HIDDEN_DIM = HEADS * DIM_HEAD

//typedef ap_fixed<16, 8> data_t;
typedef float data_t;

void vit(data_t* input_addr, data_t* output_addr);

void matmul(data_t* A_addr, int A_row, int A_col, data_t* B_addr, int B_row, int B_col,data_t* C_addr);
void matadd(data_t* A_addr,data_t* B_addr,data_t* C_addr, int row, int col);
void linear(data_t* X_addr, data_t* W_addr, data_t* B_addr, data_t* XW_addr, data_t* Rs_addr,int sample_num, int in_features, int out_features);

class str{
public:
	str();
	str(const char *s);
	int size();
	char *getptr();
	void print();
private:
	char _str[50];
	int _size;
};

class Mlp{
public:
	Mlp();
	Mlp(const char* name){
		_name = str(name);
	}
	str getName(){
		return _name;
	}
	void forward(data_t* input_addr, data_t* output_addr);
	void setW0(data_t* W_addr, int row, int col){
		for(int i = 0; i < row; i++)
			for(int j = 0; j < col; j++){
				_W0[i][j] = *(W_addr++);
			}
	}
	void setB0(data_t* B_addr, int row, int col){
		for(int i = 0; i < row; i++)
			for(int j = 0; j < col; j++){
				_B0[i][j] = *(B_addr++);
			}
	}
	void setW1(data_t* W_addr, int row, int col){
		for(int i = 0; i < row; i++)
			for(int j = 0; j < col; j++){
				_W1[i][j] = *(W_addr++);
			}
	}
	void setB1(data_t* B_addr, int row, int col){
		for(int i = 0; i < row; i++)
			for(int j = 0; j < col; j++){
				_B1[i][j] = *(B_addr++);
			}
	}
private:
	str _name;
	data_t _W0[D_MODEL][MLP_HIDDEN_DIM];
	data_t _B0[INPUT_NUM][MLP_HIDDEN_DIM];
	data_t _W1[MLP_HIDDEN_DIM][D_MODEL];
	data_t _B1[INPUT_NUM][D_MODEL];
};

class Msa{
public:
	Msa(){}
	Msa(const char* name){
		_name = str(name);
	}
	void forward(data_t* input_addr, data_t* output_addr);
	void setScale(int scale){
		_scale = scale;
	}
	void setWq(data_t* W_addr, int row, int col){
		for(int i = 0; i < row; i++)
			for(int j = 0; j < col; j++){
				_Wq[i][j] = *(W_addr++);
			}
	}
	void setWk(data_t* W_addr, int row, int col){
		for(int i = 0; i < row; i++)
			for(int j = 0; j < col; j++){
				_Wk[i][j] = *(W_addr++);
			}
	}
	void setWv(data_t* W_addr, int row, int col){
		for(int i = 0; i < row; i++)
			for(int j = 0; j < col; j++){
				_Wv[i][j] = *(W_addr++);
			}
	}
	void setWproj(data_t* W_addr, int row, int col){
		for(int i = 0; i < row; i++)
			for(int j = 0; j < col; j++){
				_Wproj[i][j] = *(W_addr++);
			}
	}
	str getName(){return _name;}
private:
	str _name;
	int _scale;
	data_t _Wq[D_MODEL][MSA_HIDDEN_DIM];
	data_t _Wk[D_MODEL][MSA_HIDDEN_DIM];
	data_t _Wv[D_MODEL][MSA_HIDDEN_DIM];
	data_t _Wproj[MSA_HIDDEN_DIM][D_MODEL];
};

class LayerNorm{
public:
	LayerNorm();
	LayerNorm(const char* name){
		_name = str(name);
	}

	void ln(data_t* X_addr, int X_row, int X_col);
	str getName(){return _name;}
private:
	data_t _gamma[10];
	data_t _beta[10];
	str _name;
};

class TransBlock{
public:
	void forward(data_t* input_addr, data_t* output_addr);
	void init();
private:
	LayerNorm ln0 = LayerNorm("LayerNorm0");
	Msa msa_block = Msa("msa_block");
	LayerNorm ln1 = LayerNorm("LayerNorm1");
	Mlp	mlp_block = Mlp("mlp_block");
};
 
#endif
