#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
#include <time.h>

#define GAP -2
#define MATCH 1
#define MISMATCH -1

#define LEFT 'a'
#define TOP 'b'
#define DIA 'c'

#define SEQ_MAX_LEN 480		// SEQ_MAX_LEN >= max length of sequence in data set
#define NUM_BLOCK 16		//block number
#define NUM_STREAM 10		//stream number
#define TASKS 3000000		//number of items from data set to process
#define SKIPS 0				//skip first N items

using namespace std;




__global__
void alignKernel(char *f_seq1, char *f_seq2, int*f_out, char *f_trace, int* f_len_reads, int* f_len_haplotypes, int max_len, int*f_score)
{
	

	char* d_seq1 = f_seq1;
	char* d_seq2 = f_seq2;
	int* e_len_reads = f_len_reads;
	int* e_len_haplotypes = f_len_haplotypes;
	int* d_out = f_out;
	char* trace = f_trace;
	int* d_score = f_score;
	
	//thread indexes
	int thread_id = threadIdx.x + blockIdx.x*blockDim.x;

	//uniformize thread_id to for easier programming
	//each block thread starts from 1
	while (thread_id >= blockDim.x){
		thread_id -= blockDim.x;
		d_seq1 += max_len;
		d_seq2 += max_len;
		e_len_reads += 1;
		e_len_haplotypes += 1;
		d_out += max_len*max_len;
		trace += max_len*max_len;
		d_score += 1;
	}

	//get length of items from reads and haplotypes in specific blocks
	int len_reads = e_len_reads[0];
	int len_haplotypes = e_len_haplotypes[0];
	
	
	int N = 0;
	if (len_reads >len_haplotypes)
		N = len_reads;
	else
		N = len_haplotypes;

	int block_size = N + 1;

	//initialize the first row and the first column of score matrix
	if (thread_id <= N){
		d_out[thread_id] = thread_id * GAP;
		d_out[thread_id *  (len_reads + 1)] = thread_id * GAP;
	}

	//threads synchronization
	__syncthreads();


	int row = 0, col = 0, left = 0, top = 0, dia = 0, max = 0, tra = 0;

	//first half matrix
	for (int i = 1; i <= block_size; i++){

		row = thread_id;
		col = i - thread_id;

		if (thread_id <= i && row > 0 && col > 0 && row <= len_haplotypes && col <= len_reads){

			//get scroes from left grid and top grid
			left = d_out[row*(len_reads + 1) + col - 1] + GAP;
			top = d_out[(row - 1)*(len_reads + 1) + col] + GAP;


			//match score
			if (d_seq2[row - 1] == d_seq1[col - 1]){
				dia = d_out[(row - 1)*(len_reads + 1) + col - 1] + MATCH;
			}

			//mismatch score
			else {				
				dia = d_out[(row - 1)*(len_reads + 1) + col - 1] + MISMATCH;				
			}

			//get final score and trace mark
			if (left > top){
				max = left;
				tra = LEFT;
			}
			else{
				max = top;
				tra = TOP;
			}

			if (dia > max){
				max = dia;
				tra = DIA;
			}
			
			//fill in score matrix and trace matrix
			d_out[row*(len_reads + 1) + col] = max;
			trace[row*(len_reads + 1) + col] = tra;

		}

		__syncthreads();

	}


	//left half matrix ; using the similar process of the first half matrix
	for (int j = 2; j <= block_size; j++){

		row = block_size - 1 - thread_id + j;
		col = thread_id;

		if (thread_id >= j - 1 && thread_id <= block_size && row > 0 && col > 0 && row <= len_haplotypes && col <= len_reads){
			
			left = d_out[row*(len_reads + 1) + col - 1] + GAP;
			top = d_out[(row - 1)*(len_reads + 1) + col] + GAP;
			
			if (d_seq2[row - 1] == d_seq1[col - 1])
				dia = d_out[(row - 1)*(len_reads + 1) + col - 1] + MATCH;
			else dia = d_out[(row - 1)*(len_reads + 1) + col - 1] + MISMATCH;

			if (left > top){
				max = left;
				tra = LEFT;
			}
			else{
				max = top;
				tra = TOP;
			}

			if (dia > max){
				max = dia;
				tra = DIA;
			}

			d_out[row*(len_reads + 1) + col] = max;
			trace[row*(len_reads + 1) + col] = tra;
		}

		__syncthreads();

	}

	//get last score
	if (thread_id == 0){
		d_score[0] = d_out[(len_reads + 1)*(len_haplotypes + 1) - 1];
	}


}


//function to communicate between host and device
void alignArray(char*seq1, char*seq2, int*len_haplotypes, int*len_reads, char*d_seq1, char*d_seq2, int*d_matrix_score, char*d_matrix_trace, int*scores, char*matrix_trace, int*d_len_reads, int*d_len_haplotypes, int max_len, int*d_score){


	//create streams
	cudaStream_t stream[NUM_STREAM];
	for (int i = 0; i < NUM_STREAM; i++)
		cudaStreamCreate(&stream[i]);

	//run streams
	for (int i = 0; i < NUM_STREAM; i++){

		//copy data from host to device
		cudaMemcpyAsync(d_seq1 + i * max_len*NUM_BLOCK, seq1 + i * max_len*NUM_BLOCK, max_len*NUM_BLOCK*sizeof(char), cudaMemcpyHostToDevice, stream[i]);
		cudaMemcpyAsync(d_seq2 + i * max_len*NUM_BLOCK, seq2 + i * max_len*NUM_BLOCK, max_len*NUM_BLOCK*sizeof(char), cudaMemcpyHostToDevice, stream[i]);
		cudaMemcpyAsync(d_len_reads + i * NUM_BLOCK, len_reads + i * NUM_BLOCK, NUM_BLOCK*sizeof(int), cudaMemcpyHostToDevice, stream[i]);
		cudaMemcpyAsync(d_len_haplotypes + i * NUM_BLOCK, len_haplotypes + i * NUM_BLOCK, NUM_BLOCK*sizeof(int), cudaMemcpyHostToDevice, stream[i]);
	
		//run kernel
		alignKernel << <NUM_BLOCK, max_len, 0, stream[i] >> >(d_seq1 + i * max_len*NUM_BLOCK, d_seq2 + i * max_len*NUM_BLOCK, d_matrix_score + i * max_len* max_len*NUM_BLOCK, d_matrix_trace + i * max_len* max_len*NUM_BLOCK, d_len_reads + i*NUM_BLOCK, d_len_haplotypes + i*NUM_BLOCK, max_len, d_score + i*NUM_BLOCK);

		//copy back
		cudaMemcpyAsync(scores + i *NUM_BLOCK, d_score + i*NUM_BLOCK, NUM_BLOCK*sizeof(int), cudaMemcpyDeviceToHost, stream[i]);
		cudaMemcpyAsync(matrix_trace + i *max_len* max_len*NUM_BLOCK, d_matrix_trace + i *max_len* max_len*NUM_BLOCK, max_len* max_len*NUM_BLOCK*sizeof(char), cudaMemcpyDeviceToHost, stream[i]);
		
	}

	// destroy streams
	for (int i = 0; i < NUM_STREAM; ++i)
		cudaStreamDestroy(stream[i]);

}


//big loop to read in data from file
void loop(string *reads, string *haplotypes, int* d_matrix_score, char* d_seq1, char* d_seq2, char* d_matrix_trace, int*d_len_reads, int* d_len_haplotypes, int*d_score, ofstream &outf){
	
	
	int num_data = NUM_BLOCK*NUM_STREAM;		//number of data in one big loop
	int *len_reads = (int*)calloc(num_data, sizeof(int));
	int *len_haplotypes = (int*)calloc(num_data, sizeof(int));

	//get max length of data in one loop for unification malloc
	int max_len = 0;

	for (int i = 0; i < num_data; i++){
		len_reads[i] = reads[i].length();
		len_haplotypes[i] = haplotypes[i].length();
		if (len_reads[i] > max_len)
			max_len = len_reads[i];
		if (len_haplotypes[i] > max_len)
			max_len = len_haplotypes[i];
	}
	max_len += 1;

	//sequences from host to the device
	char*seq1, *seq2;
	cudaHostAlloc(&seq1, max_len * num_data* sizeof(char), cudaHostAllocDefault);
	cudaHostAlloc(&seq2, max_len * num_data* sizeof(char), cudaHostAllocDefault);

	//scores and trace matrix 
	int *scores = (int*)calloc(num_data, sizeof(int));
	char *matrix_trace = (char*)calloc(num_data*max_len*max_len, sizeof(char));
	

	//copy items from file to seq1 and seq2
	for (int i = 0; i < num_data; i++){

		strcpy(seq1 + max_len * i, haplotypes[i].c_str());
		strcpy(seq2 + max_len * i, reads[i].c_str());

	}

	//cuda function
	alignArray(seq1, seq2, len_reads, len_haplotypes, d_seq1, d_seq2, d_matrix_score, d_matrix_trace, scores, matrix_trace, d_len_reads, d_len_haplotypes, max_len, d_score);




	//output results

	for (int i = 0; i < num_data; i++){

		//get output sequence
		int row = len_reads[i];
		int col = len_haplotypes[i];
		int t_seq = 0;
		char *trace_seq = (char*)calloc(row + col, sizeof(char));

		//get trace sequence array
		while (row > 0 && col >0){

			trace_seq[t_seq] = matrix_trace[max_len*max_len*i + row*(len_haplotypes[i] + 1) + col];
			if (trace_seq[t_seq] == DIA){
				row--;
				col--;
			}
			else if (trace_seq[t_seq] == LEFT){
				col--;
			}
			else if (trace_seq[t_seq == TOP]){
				row--;
			}
			t_seq++;

		}

		while (row > 0){
			trace_seq[t_seq] = TOP;
			row--;
			t_seq++;
		}
		while (col > 0){
			trace_seq[t_seq] = LEFT;
			col--;
			t_seq++;
		}
		

		int output_len = t_seq;

		char *out_reads = new char[output_len];
		char *out_haplotypes = new char[output_len];


		//output haplotype

		int haplotypes_out = 0;

		for (int t_out = 0; t_out < output_len; t_out++){
			if (trace_seq[output_len - t_out - 1] == DIA){
				
				out_haplotypes[t_out] = seq1[haplotypes_out + i*max_len];
				haplotypes_out++;

			}
			else if (trace_seq[output_len - t_out - 1] == TOP){
				
				out_haplotypes[t_out] = '_';

			}
			else if (trace_seq[output_len - t_out - 1] == LEFT){
				
				out_haplotypes[t_out] = seq1[haplotypes_out + i*max_len];
				haplotypes_out++;

			}
		}

		for (int b = 0; b < output_len; b++){

			outf << out_haplotypes[b];

		}

		outf << endl;

		//output score
		outf << scores[i] << endl;

		//output read
		int read_out = 0;
		for (int t_out = 0; t_out < output_len; t_out++){
			if (trace_seq[output_len - t_out - 1] == DIA){

				out_reads[t_out] = seq2[read_out + i*max_len];
				read_out++;

			}
			else if (trace_seq[output_len - t_out - 1] == TOP){

				out_reads[t_out] = seq2[read_out + i*max_len];
				read_out++;

			}
			else if (trace_seq[output_len - t_out - 1] == LEFT){

				out_reads[t_out] = '_';

			}
		}

		for (int b = 0; b < output_len; b++){

			outf << out_reads[b];

		}

		outf << endl;
		outf << endl;


		delete out_reads;
		delete out_haplotypes;
		free(trace_seq);


		//display results in screen
		//cout << endl;
		//cout << "****************************************************************************************************" << endl;

		//for (int x = 0; x < len_reads[i] + 1; x++){
		//	for (int y = 0; y < len_haplotypes[i] + 1; y++){
		//		cout << matrix_trace[i*max_len*max_len + x*(len_haplotypes[i] + 1) + y];
		//		cout << "\t";
		//	}
		//	cout << endl;
		//}

		//cout << "x=" << i << endl;	//number of rows in files

		//cout << "     reads:" << reads[i] << endl;
		//cout << "haplotypes:" << haplotypes[i] << endl;
		//cout << "length of reads:" << len_reads[i] << endl;
		//cout << "length of haplotypes:" << len_haplotypes[i] << endl;

		//cout << "score" << i << ":" << scores[i] << endl;

		//cout << "trace_seq:";
		//for (int x = 0; x < t_seq; x++){
		//	cout << trace_seq[x];
		//}
		//cout << endl;

		//cout << "row:" << row << endl;
		//cout << "col:" << col << endl;

		//cout << "len_output_seq:" << t_seq<<endl;

		//cout << "haplotypes_out:";
		//for (int b = 0; b < output_len; b++){
		//	cout << out_haplotypes[b];
		//}
		//cout<<endl;

		//cout << "      read_out:";
		//for (int b = 0; b < output_len; b++){
		//	cout << out_reads[b];
		//}
		//cout<<endl;

		//cout << "****************************************************************************************************" << endl;

		//cout << endl;


	}

	
	//free memory

	free(len_reads);
	free(len_haplotypes);
	free(scores);
	free(matrix_trace);
	
	cudaFreeHost(seq1);
	cudaFreeHost(seq2);


}





int main()
{
	//open files
	ifstream readsfile("H:/Hust/大三下/精准医疗信息导论/project/GPU/数据/reads.txt");						//reads file
	ifstream haplotypesfile("H:/Hust/大三下/精准医疗信息导论/project/GPU/数据/haplotypes.txt");			//haplotypes file
	ofstream outf("H:/Hust/大三下/精准医疗信息导论/project/GPU/数据/gpu_output/AlignResults.txt");			//output file
	ofstream timef("H:/Hust/大三下/精准医疗信息导论/project/GPU/数据/gpu_output/demo05/time.txt", ios::app);	//time record file


	//ifstream readsfile("H:/Hust/大三下/精准医疗信息导论/project/GPU/测试数据/reads.txt");
	//ifstream haplotypesfile("H:/Hust/大三下/精准医疗信息导论/project/GPU/测试数据/haplotypes.txt");
	//ofstream outf("H:/Hust/大三下/精准医疗信息导论/project/GPU/测试数据/AlignResults.txt");
	//ofstream timef("H:/Hust/大三下/精准医疗信息导论/project/GPU/测试数据/gpu_output/time.txt", ios::app);


	//if fail to open files, break
	assert(readsfile.is_open());
	assert(haplotypesfile.is_open());
	assert(outf.is_open);
	assert(timef.is_open);


	string skip_reads, skip_hap;
	for (int c = 0; c < SKIPS; c++){
		getline(readsfile, skip_reads);
		getline(haplotypesfile, skip_hap);
	}

	//program running time
	double start, stop, durationTime;
	start = clock();
	
	
	int loop_main = int(TASKS / (NUM_STREAM*NUM_BLOCK));



	//malloc GPU

	int *d_matrix_score = 0;
	char *d_seq1 = 0;
	char *d_seq2 = 0;
	char *d_matrix_trace = 0;
	int *d_len_reads = 0;
	int *d_len_haplotypes = 0;
	int *d_score = 0;

	cudaMalloc(&d_matrix_score, (SEQ_MAX_LEN + 1) * (SEQ_MAX_LEN + 1) * NUM_BLOCK* NUM_STREAM * sizeof(int));
	cudaMalloc(&d_seq1, (SEQ_MAX_LEN + 1) * NUM_BLOCK * NUM_STREAM * sizeof(char));
	cudaMalloc(&d_seq2, (SEQ_MAX_LEN + 1) * NUM_BLOCK * NUM_STREAM * sizeof(char));
	cudaMalloc(&d_matrix_trace, (SEQ_MAX_LEN + 1) * (SEQ_MAX_LEN + 1) * NUM_BLOCK * NUM_STREAM * sizeof(char));
	cudaMalloc(&d_len_reads,  NUM_BLOCK * NUM_STREAM * sizeof(int));
	cudaMalloc(&d_len_haplotypes, NUM_BLOCK * NUM_STREAM * sizeof(int));
	cudaMalloc(&d_score, NUM_BLOCK * NUM_STREAM * sizeof(int));



	for (int x = 0; x < loop_main; x++){

		string reads[NUM_STREAM*NUM_BLOCK];
		string haplotypes[NUM_STREAM*NUM_BLOCK];
		for (int a = 0; a < NUM_STREAM*NUM_BLOCK; a++){
			getline(readsfile, reads[a]);
			getline(haplotypesfile, haplotypes[a]);
		}


		//show rate of progress
		cout << "loop:" << x+1 << endl;
		system("cls");
		
		//strat one loop to process NUM_STREAM*NUM_BLOCK items at the same time
		loop(reads, haplotypes, d_matrix_score, d_seq1, d_seq2, d_matrix_trace, d_len_reads, d_len_haplotypes, d_score, outf);


	}

	stop = clock();

	durationTime = ((double)(stop - start)) / CLK_TCK;
	cout << "time:" << durationTime << endl;


	//free memory
	cudaFree(d_matrix_score);
	cudaFree(d_seq1);
	cudaFree(d_seq2);
	cudaFree(d_matrix_trace);
	cudaFree(d_len_reads);
	cudaFree(d_len_haplotypes);
	cudaFree(d_score);

	//output time to time record file
	timef << endl <<TASKS << ":" << durationTime;

	//close files
	readsfile.close();
	haplotypesfile.close();
	outf.close();
	timef.close();
	return 0;


}






