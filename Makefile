# This Makefile assumes the following module files are loaded:
#
# CUDA
#
# This Makefile will only work if executed on a GPU node.
#

NVCC = nvcc

NVCCFLAGS = -O3 -g

LFLAGS = -lm -g

# Compiler-specific flags (by default, we always use sm_75)
GENCODE_SM75 = -gencode=arch=compute_75,code=sm_75
GENCODE = $(GENCODE_SM75)

.SUFFIXES : .cu

BINARIES = matmul recmatmul tiledmatmul adjmatmul

matmul: matmul.o
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $<

recmatmul: recmatmul.o
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $<

tiledmatmul: tiledmatmul.o
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $<

adjmatmul: adjmatmul.o
	$(NVCC) $(GENCODE) $(LFLAGS) -o $@ $<

.cu.o:
	$(NVCC) $(GENCODE) $(NVCCFLAGS) -o $@ -c $<

clean:	
	rm -f *.o $(BINARIES)
