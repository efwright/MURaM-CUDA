#CUDA_PATH=/glade/u/apps/dav/opt/cuda/11.4.0

#rm rt_integrate

#nvcc -c integrate.cu -O3 -std=c++11
#nvcc -c CG/cg.cu -O3 -std=c++11 -gencode arch=compute_70,code=sm_70 -I$CUDA_PATH/include -L$CUDA_PATH/lib64 -lcudart
#nvcc -c Graphs/graph.cu -O3 -std=c++11
#nvc++ -o rt_integrate main.cpp oacc.cpp integrate.o cg.o graph.o -gpu=cc70 -acc -Minfo=all -O3 -std=c++11 -I$CUDA_PATH/include -L$CUDA_PATH/lib64 -lcudart

CUDA_ROOT=/glade/u/apps/dav/opt/cuda/11.4.0

###############################################
# OpenACC Compiler Options
CXX=nvc++
CXX_FLAGS=-O3 -std=c++11 -gpu=cc70 -acc -Minfo=all

###############################################
# NVCC Compiler Options
NVCC=nvcc
NVCC_FLAGS=-O3 -std=c++11

CUDA_LIB=-L$(CUDA_ROOT)/lib64 -lcudart
CUDA_INC=-I$(CUDA_ROOT)/include

###############################################
SRC_DIR=src
OBJ_DIR=bin

RT_OBJS=$(OBJ_DIR)/main.o $(OBJ_DIR)/oacc.o $(OBJ_DIR)/integrate.o $(OBJ_DIR)/cg.o $(OBJ_DIR)/graph.o

###############################################
rt_integrate: $(RT_OBJS)
	$(CXX) $(CXX_FLAGS) $(RT_OBJS) -o $@ $(CUDA_INC) $(CUDA_LIB)

$(OBJ_DIR)/%.o: %.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

clean:
	$(RM) bin/* *.o rt_integrate

