#ifndef __VIT__
#define __VIT__

#define PI 3.1415926
#define D_MODLE 10
#define INPUT_NUM 20
#define MLP_NUM 2
#define MLP_IN_FEATURES 4
#define MLP_HIDDEN_DIM 3
#define MLP_OUT_FEATURES 2

#define MSA_NUM 2
#define MSA_IN_FEATURES 4
#define MSA_W_DIM 3
#define MSA_OUT_FEATURES 2

typedef float data_t;

void vit();

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
	data_t _W0[MLP_IN_FEATURES][MLP_HIDDEN_DIM];
	data_t _B0[MLP_NUM][MLP_HIDDEN_DIM];
	data_t _W1[MLP_HIDDEN_DIM][MLP_OUT_FEATURES];
	data_t _B1[MLP_NUM][MLP_OUT_FEATURES];
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
private:
	str _name;
	int _scale;
	data_t _Wq[MSA_IN_FEATURES][MSA_W_DIM];
	data_t _Wk[MSA_IN_FEATURES][MSA_W_DIM];
	data_t _Wv[MSA_IN_FEATURES][MSA_W_DIM];
	data_t _Wproj[MSA_W_DIM][MSA_OUT_FEATURES];
};

class LayerNorm{
public:
	LayerNorm();
	LayerNorm(const char* name){
		_name = str(name);
	}

	void ln(data_t* X_addr, int X_row, int X_col);
private:
	data_t _gamma[10];
	data_t _beta[10];
	str _name;
};

class TransBlock{
public:
	void forward(data_t* input_addr, data_t* output_addr);
private:
};
 
#endif
