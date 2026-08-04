GGML = /home/lhao16/ggml

CXXFLAGS = -std=c++17 -O2 -I $(GGML)/include
LDFLAGS  = -L $(GGML)/build/src -lggml -lggml-base -lggml-cpu \
           -Wl,-rpath,$(GGML)/build/src -lm -lpthread

all: test_iris run_iris

test_iris: test_iris.cpp iris.cpp iris.h
	g++ $(CXXFLAGS) test_iris.cpp iris.cpp -o $@ $(LDFLAGS)

run_iris: run_iris.cpp iris.cpp iris.h
	g++ $(CXXFLAGS) run_iris.cpp iris.cpp -o $@ $(LDFLAGS)

clean:
	rm -f test_iris run_iris
