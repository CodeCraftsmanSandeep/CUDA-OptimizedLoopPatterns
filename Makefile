# Compiler and flags
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

# Thrust
Thrust:
	$(NVCC) $(CFLAGS) State-Of-The-Art-Comparision/thrustReduction.cu -o State-Of-The-Art-Comparision/thrustReduction.out

# CUB
CUB:
	$(NVCC) $(CFLAGS) State-Of-The-Art-Comparision/CUBReduction.cu -o State-Of-The-Art-Comparision/CUBReduction.out

# Build all
all: BitwiseXOR Addition GCD Min Thrust CUB

# Clean
clean:
	rm -f Reduction/BitwiseXor/*.out
	rm -f Reduction/Addition/*.out
	rm -f Reduction/GCD/*.out
	rm -f Reduction/Min/*.out
	rm -f State-Of-The-Art-Comparision/*.out

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

