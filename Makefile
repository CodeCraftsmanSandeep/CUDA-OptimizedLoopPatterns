#Compiler and flags
NVCC = nvcc
CFLAGS = -O3 -arch=sm_75 -Iinclude/

INPUT_DIR = input

# Targets for BitwiseXOR
BitwiseXOR: Reduction/BitwiseXor/src.cu
	$(NVCC) $(CFLAGS) Reduction/BitwiseXor/src.cu Reduction/BitwiseXor/gridStride.cu -o Reduction/BitwiseXor/gridStride.out
	$(NVCC) $(CFLAGS) Reduction/BitwiseXor/src.cu Reduction/BitwiseXor/blockStride.cu -o Reduction/BitwiseXor/blockStride.out
	$(NVCC) $(CFLAGS) Reduction/BitwiseXor/src.cu Reduction/BitwiseXor/warpStride.cu -o Reduction/BitwiseXor/warpStride.out

# Targets for Addition
Addition: Reduction/Addition/src.cu
	$(NVCC) $(CFLAGS) Reduction/Addition/src.cu Reduction/Addition/gridStride.cu -o Reduction/Addition/gridStride.out
	$(NVCC) $(CFLAGS) Reduction/Addition/src.cu Reduction/Addition/blockStride.cu -o Reduction/Addition/blockStride.out
	$(NVCC) $(CFLAGS) Reduction/Addition/src.cu Reduction/Addition/warpStride.cu -o Reduction/Addition/warpStride.out

# Targets for GCD
GCD: Reduction/GCD/src.cu
	$(NVCC) $(CFLAGS) Reduction/GCD/src.cu Reduction/GCD/gridStride.cu -o Reduction/GCD/gridStride.out
	$(NVCC) $(CFLAGS) Reduction/GCD/src.cu Reduction/GCD/blockStride.cu -o Reduction/GCD/blockStride.out
	$(NVCC) $(CFLAGS) Reduction/GCD/src.cu Reduction/GCD/warpStride.cu -o Reduction/GCD/warpStride.out

# Targets for Min
Min: Reduction/Min/src.cu
	$(NVCC) $(CFLAGS) Reduction/Min/src.cu Reduction/Min/gridStride.cu -o Reduction/Min/gridStride.out
	$(NVCC) $(CFLAGS) Reduction/Min/src.cu Reduction/Min/blockStride.cu -o Reduction/Min/blockStride.out
	$(NVCC) $(CFLAGS) Reduction/Min/src.cu Reduction/Min/warpStride.cu -o Reduction/Min/warpStride.out

thrustMin:
	$(NVCC) $(CFLAGS) Reduction/Min/thrustMin.cu -o Reduction/Min/thrustMin.out

CUBMin:
	$(NVCC) $(CFLAGS) Reduction/Min/CUBMin.cu -o Reduction/Min/CUBMin.out

thrustGCD:
	$(NVCC) $(CFLAGS) Reduction/GCD/thrustGCD.cu -o Reduction/GCD/thrustGCD.out

CUBGCD:
	$(NVCC) $(CFLAGS) Reduction/GCD/CUBGCD.cu -o Reduction/GCD/CUBGCD.out

thrustXOR:
	$(NVCC) $(CFLAGS) Reduction/BitwiseXor/thrustXOR.cu -o Reduction/BitwiseXor/thrustXOR.out

CUBXOR:
	$(NVCC) $(CFLAGS) Reduction/BitwiseXor/CUBXOR.cu -o Reduction/BitwiseXor/CUBXOR.out

DotProduct: DotProductGridStride DotProductBlockStride DotProductWarpStride thrustDotProduct CUBDotProduct

DotProductGridStride:
	$(NVCC) $(CFLAGS) DotProduct/src.cu DotProduct/gridStride.cu -o DotProduct/gridStride.out

DotProductBlockStride:
	$(NVCC) $(CFLAGS) DotProduct/src.cu DotProduct/blockStride.cu -o DotProduct/blockStride.out

DotProductWarpStride:
	$(NVCC) $(CFLAGS) DotProduct/src.cu DotProduct/warpStride.cu -o DotProduct/warpStride.out

thrustDotProduct:
	$(NVCC) $(CFLAGS) DotProduct/thrustDotProduct.cu -o DotProduct/thrustDotProduct.out

CUBDotProduct:
	$(NVCC) $(CFLAGS) DotProduct/CUBDotProduct.cu -o DotProduct/CUBDotProduct.out

L1Norm: L1NormGridStride L1NormBlockStride L1NormWarpStride thrustL1Norm CUBL1Norm

L1NormGridStride:
	$(NVCC) $(CFLAGS) L1Norm/src.cu L1Norm/gridStride.cu -o L1Norm/gridStride.out

L1NormBlockStride:
	$(NVCC) $(CFLAGS) L1Norm/src.cu L1Norm/blockStride.cu -o L1Norm/blockStride.out

L1NormWarpStride:
	$(NVCC) $(CFLAGS) L1Norm/src.cu L1Norm/warpStride.cu -o L1Norm/warpStride.out

thrustL1Norm:
	$(NVCC) $(CFLAGS) L1Norm/thrustL1Norm.cu -o L1Norm/thrustL1Norm.out

CUBL1Norm:
	$(NVCC) $(CFLAGS) L1Norm/CUBL1Norm.cu -o L1Norm/CUBL1Norm.out

L2Norm: L2NormGridStride L2NormBlockStride L2NormWarpStride

L2NormGridStride:
	$(NVCC) $(CFLAGS) L2Norm/src.cu L2Norm/gridStride.cu -o L2Norm/gridStride.out

L2NormBlockStride:
	$(NVCC) $(CFLAGS) L2Norm/src.cu L2Norm/blockStride.cu -o L2Norm/blockStride.out

L2NormWarpStride:
	$(NVCC) $(CFLAGS) L2Norm/src.cu L2Norm/warpStride.cu -o L2Norm/warpStride.out

thrustL2Norm:
	$(NVCC) $(CFLAGS) L2Norm/thrustL2Norm.cu -o L2Norm/thrustL2Norm.out

CUBL2Norm:
	$(NVCC) $(CFLAGS) L2Norm/CUBL2Norm.cu -o L2Norm/CUBL2Norm.out

LInfNorm: LInfNormGridStride LInfNormBlockStride LInfNormWarpStride

LInfNormGridStride:
	$(NVCC) $(CFLAGS) LInfNorm/src.cu LInfNorm/gridStride.cu -o LInfNorm/gridStride.out

LInfNormBlockStride:
	$(NVCC) $(CFLAGS) LInfNorm/src.cu LInfNorm/blockStride.cu -o LInfNorm/blockStride.out

LInfNormWarpStride:
	$(NVCC) $(CFLAGS) LInfNorm/src.cu LInfNorm/warpStride.cu -o LInfNorm/warpStride.out

thrustLInfNorm:
	$(NVCC) $(CFLAGS) LInfNorm/thrustLInfNorm.cu -o LInfNorm/thrustLInfNorm.out

CUBLInfNorm:
	$(NVCC) $(CFLAGS) LInfNorm/CUBLInfNorm.cu -o LInfNorm/CUBLInfNorm.out

# Thrust
Thrust:
	$(NVCC) $(CFLAGS)cccl/ State-Of-The-Art-Comparision/thrustReduction/thrustReduction.cu -o State-Of-The-Art-Comparision/thrustReduction/thrustReduction.out

# CUB
CUB:
	$(NVCC) $(CFLAGS)cccl/ State-Of-The-Art-Comparision/CUBReduction/CUBReduction.cu -o State-Of-The-Art-Comparision/CUBReduction/CUBReduction.out

# Build all
all: BitwiseXOR Addition GCD Min Thrust CUB

# Clean
clean:
	rm -f Reduction/BitwiseXor/*.out
	rm -f Reduction/Addition/*.out
	rm -f Reduction/GCD/*.out
	rm -f Reduction/Min/*.out
	rm -f State-Of-The-Art-Comparision/thrustReduction/*.out
	rm -f State-Of-The-Art-Comparision/CUBReduction/*.out

# Run
run:
	@if [ -z "$(flags)" ] || echo "$(flags)" | grep -q -- "--BitwiseXOR"; then \
		echo "▶ Running BitwiseXOR experiments..."; \
		bash ./runExperiments.sh Reduction/BitwiseXor $(INPUT_DIR) results/BitwiseXOR; \
	fi; \
	if [ -z "$(flags)" ] || echo "$(flags)" | grep -q -- "--Addition"; then \
		echo "▶ Running Addition experiments..."; \
		bash ./runExperiments.sh Reduction/Addition $(INPUT_DIR) results/Addition; \
	fi; \
	if echo "$(flags)" | grep -q -- "--GCD"; then \
		echo "▶ Running GCD experiments..."; \
		bash ./runExperiments.sh Reduction/GCD $(INPUT_DIR) results/GCD; \
	fi; \
	if echo "$(flags)" | grep -q -- "--Min"; then \
		echo "▶ Running Min experiments..."; \
		bash ./runExperiments.sh Reduction/Min $(INPUT_DIR) results/Min; \
	fi; \
	if echo "$(flags)" | grep -q -- "--Thrust"; then \
		echo "▶ Running Thrust experiments..."; \
		bash ./runExperiments.sh State-Of-The-Art-Comparision/thrustReduction $(INPUT_DIR) results/Thrust; \
	fi; \
	if echo "$(flags)" | grep -q -- "--CUB"; then \
		echo "▶ Running CUB experiments..."; \
		bash ./runExperiments.sh State-Of-The-Art-Comparision/CUBReduction $(INPUT_DIR) results/CUB; \
	fi

