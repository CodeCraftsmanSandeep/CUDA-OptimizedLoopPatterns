BitwiseXOR: Reduction/BitwiseXor/src.cu
	 nvcc -O3 -arch=sm_75 -Iinclude/ Reduction/BitwiseXor/src.cu Reduction/BitwiseXor/gridStride.cu -o Reduction/BitwiseXor/gridStride.out
	 nvcc -O3 -arch=sm_75 -Iinclude/ Reduction/BitwiseXor/src.cu Reduction/BitwiseXor/blockStride.cu -o Reduction/BitwiseXor/blockStride.out
	 nvcc -O3 -arch=sm_75 -Iinclude/ Reduction/BitwiseXor/src.cu Reduction/BitwiseXor/warpStride.cu -o Reduction/BitwiseXor/warpStride.out
