CC = mpicxx

default: knn

knn: pr3.cpp
	${CC} -O3 -fopenmp -o $@ pr3.cpp

clean:
	-rm -f knn
