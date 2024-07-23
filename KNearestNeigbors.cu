#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <stdexcept>
#include <iostream>
#include <fstream>


#define BLOCK_SIZE 16

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

using idx_t = std::uint32_t;
using val_t = float;
using ptr_t = std::uintptr_t;

typedef struct csr_t {
  idx_t nrows; // number of rows
  idx_t ncols; // number of rows
  idx_t * ind; // column ids
  val_t * val; // values
  ptr_t * ptr; // pointers (start of row in ind/val)

  csr_t()
  {
    nrows = ncols = 0;
    ind = nullptr;
    val = nullptr;
    ptr = nullptr;
  }

  void reserve(const idx_t nrows, const ptr_t nnz)
  {
    if(nrows > this->nrows){
      if(ptr){
        ptr = (ptr_t*) realloc(ptr, sizeof(ptr_t) * (nrows+1));
      } else {
        ptr = (ptr_t*) malloc(sizeof(ptr_t) * (nrows+1));
        ptr[0] = 0;
      }
      if(!ptr){
        throw std::runtime_error("Could not allocate ptr array.");
      }
    }
    if(ind){
      ind = (idx_t*) realloc(ind, sizeof(idx_t) * nnz);
    } else {
      ind = (idx_t*) malloc(sizeof(idx_t) * nnz);
    }
    if(!ind){
      throw std::runtime_error("Could not allocate ind array.");
    }
    if(val){
      val = (val_t*) realloc(val, sizeof(val_t) * nnz);
    } else {
      val = (val_t*) malloc(sizeof(val_t) * nnz);
    }
    if(!val){
      throw std::runtime_error("Could not allocate val array.");
    }
    this->nrows = nrows;
  }

  void read(const std::string &filename)
  {
    FILE * infile = fopen(filename.c_str(), "r");
    char * line = NULL;
    size_t n, nr, nnz;
    char *head;
    char *tail;
    idx_t cid;
    double dval;
    
    if (!infile) {
      throw std::runtime_error("Could not open CLU file\n");
    }
    if(getline (&line, &n, infile) < 0){
      throw std::runtime_error("Could not read first line from CLU file\n");
    }
    //read matriz size info
    size_t rnrows, rncols, rnnz;
    sscanf(line, "%zu %zu %zu", &rnrows, &rncols, &rnnz);

    //allocate space
    this->reserve(rnrows, rnnz);
    ncols = rncols;
    
    //read in rowval, rowind, rowptr
    this->ptr[0]= 0;
    nnz = 0;
    nr = 0;

    while(getline(&line, &n, infile) != -1){
      head = line;
      while (1) {
        cid = (idx_t) strtol(head, &tail, 0);
        if (tail == head)
          break;
        head = tail;

        if(cid <= 0){
          throw std::runtime_error("Invalid column ID while reading CLUTO matrix\n");
        }
        this->ind[nnz] = cid - 1; //csr/clu files have 1-index based column IDs and our matrix is 0-based.
        dval = strtod(head, &tail);
        head = tail;
        this->val[nnz++] = dval;
      }
      this->ptr[nr+1] = nnz;
      nr++;
    }
    assert(nr == rnrows);
    free(line);
    fclose(infile);
  }

  static csr_t * from_CLUTO(const std::string &filename)
  {
    auto mat = new csr_t();
    mat->read(filename);
    return mat;
  }

  void write(const std::string output_fpath, const bool header=false)
  {
    std::fstream resfile;
    resfile.open(output_fpath, std::ios::out);
    if(!resfile){
      throw std::runtime_error("Could not open output file for writing.");
    }
    if(header){
      resfile << nrows << " " << ncols << " " << ptr[nrows] << std::endl;
    }
    for(idx_t i=0; i < nrows; ++i){
      for(ptr_t j=ptr[i]; j < ptr[i+1]; ++j){
        resfile << ind[j] << " " << val[j];
        if(j+1 < ptr[i+1]){
          resfile << " ";
        }
      }
      resfile << std::endl;
    }
    resfile.close();
  }

  __global__ void normalize(int norm=2)
  {
    val_t sum;
    //for (idx_t i = 0; i < nrows; i++) { // each row
    int i = threadIdx.x;
    sum = 0;
    for (ptr_t j = ptr[i]; j < ptr[i + 1]; j++) { // each value in row
        if (norm == 2) {
        sum += val[j] * val[j];
        } else if (norm == 1) {
        sum += val[j] > 0 ? val[j] : -val[j];
        } else {
        throw std::runtime_error("Norm must be 1 or 2.");
        }
    }
    if (sum > 0) {
        if (norm == 2) {
        sum = (double) 1.0 / sqrt(sum);
        } else {
        sum = (double) 1.0 / sum;
        }
        for (ptr_t j = ptr[i]; j < ptr[i + 1]; j++) {
        val[j] *= sum;
        }
    }
  }

  ~csr_t()
  {
    if(ind){
      free(ind);
    }
    if(val){
      free(val);
    }
    if(ptr){
      free(ptr);
    }
  }
} csr_t;