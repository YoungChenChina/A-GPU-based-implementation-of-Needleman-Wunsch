This is a GPU based implementation of Needleman-Wunsch algrithm for global sequence alignment.

Before running demo, you need to change the imput data path in the code.

When setting the number of sequences for alignment, it is better to fit the times of NUM_STREAM*NUM_BLOCK, if not, the rest sequences will be ignored and need to be comput after that, which can be achieved without changing the data set using SKPIS to skip the sequences already done.

When initializing, SEQ_MAX_LEN need to be set to initialize the ram in GPU, which should be larger than the minimum length of sequences for alignment.

Number of stream used in CUDA can be set by seting NUM_STREAM.
Number of blocks per stream can also be set by NUM_BLOCK.
Number of sequences skiped by required can be set by SKIPS.
