#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <cstring>      /* strcasecmp */
#include <cstdint>
#include <assert.h>
#include <vector>       // std::vector
#include <algorithm>    // std::random_shuffle
#include <random>
#include <stdexcept>
#include <algorithm>    // std::min_element, std::max_element
#include <array>
#include <mpi.h>

using namespace std;

using idx_t = std::uint32_t;
using val_t = float;
using ptr_t = std::uintptr_t;

/**
 * CSR structure to store search results
 */
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

  /**
   * Reserve space for more rows or non-zeros. Structure may only grow, not shrink.
   * @param nrows Number of rows
   * @param nnz   Number of non-zeros
   */
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

  csr_t ( const csr_t &other)
  {
    this->nrows = this->ncols = 0;
    this->ptr = nullptr;
    this->ind = nullptr;
    this->val = nullptr;
    this->reserve(other.nrows, other.ptr[other.nrows]);
    memcpy(ptr, other.ptr, sizeof(ptr_t) * (nrows+1));
    memcpy(ind, other.ind, sizeof(idx_t) * ptr[nrows]);
    memcpy(val, other.val, sizeof(val_t) * ptr[nrows]);
    this->ncols = other.ncols;
  }

  /**
   * Create random matrix with given sparsity factor.
   * @param nrows Number of rows
   * @param ncols Number of columns
   * @param factor   Sparsity factor
   */
  static csr_t * random(const idx_t nrows, const idx_t ncols, const double factor)
  {
    ptr_t nnz = (ptr_t) (factor * nrows * ncols);
    if(nnz >= nrows * ncols / 2.0){
      throw std::runtime_error("Asking for too many non-zeros. Matrix is not sparse.");
    }
    auto mat = new csr_t();
    mat->reserve(nrows, nnz);
    mat->ncols = ncols;

    /* fill in ptr array; generate random row sizes */
    unsigned int seed = (unsigned long) mat;
    long double sum = 0;
    for(idx_t i=1; i <= mat->nrows; ++i){
      mat->ptr[i] = rand_r(&seed) % ncols;
      sum += mat->ptr[i];
    }
    for(idx_t i=0; i < mat->nrows; ++i){
      double percent = mat->ptr[i+1] / sum;
      mat->ptr[i+1] = mat->ptr[i] + (ptr_t)(percent * nnz);
      if(mat->ptr[i+1] > nnz){
        mat->ptr[i+1] = nnz;
      }
    }
    if(nnz - mat->ptr[mat->nrows-1] <= ncols){
      mat->ptr[mat->nrows] = nnz;
    }

    /* fill in indices and values with random numbers */
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      unsigned int seed = (unsigned long) mat * (1+tid);
      std::vector<int> perm;
      for(idx_t i=0; i < ncols; ++i){
        perm.push_back(i);
      }
      std::random_device seeder;
      std::mt19937 engine(seeder());

      #pragma omp for
      for(idx_t i=0; i < nrows; ++i){
        std::shuffle(perm.begin(), perm.end(), engine);
        for(ptr_t j=mat->ptr[i]; j < mat->ptr[i+1]; ++j){
          mat->ind[j] = perm[j - mat->ptr[i]];
          mat->val[j] = ((double) rand_r(&seed)/rand_r(&seed));
        }
      }
    }

    return mat;
  }

  string info(const string name="") const
  {
    return (name.empty() ? "CSR" : name) + "<" + to_string(nrows) + ", " + to_string(ncols) + ", " +
      (ptr ? to_string(ptr[nrows]) : "0") + ">";
  }

  /** 
   * Read the matrix from a CLUTO file.
   * The first line is "nrows ncols nnz".
   * Each other line is a row in the matrix containing column ID-value pairs for non-zeros in the row.
  */
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
  
  /** 
   * Read the matrix from a CLUTO file.
   * The first line is "nrows ncols nnz".
   * Each other line is a row in the matrix containing column ID-value pairs for non-zeros in the row.
  */
  static csr_t * from_CLUTO(const std::string &filename)
  {
    auto mat = new csr_t();
    mat->read(filename);
    return mat;
  }

  /**
   * Write matrix to text file
   * @param output_fpath File to write to
   */
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

  /**
   * Normalize the rows of the matrix by L1 or L2 norm.
   * @param norm The norm of the matrix to normalize by.
  */
  void normalize(int norm=2)
  {
    #pragma omp parallel if (ptr[nrows] > 1e+6)
    {
      val_t sum;
      #pragma omp for schedule (static)
      for (idx_t i = 0; i < nrows; i++) { // each row
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

/**
 * Ensure the matrix is valid
 * @param mat Matrix to test
 */
void test_matrix(csr_t * mat){
  auto nrows = mat->nrows;
  auto ncols = mat->ncols;
  assert(mat->ptr);
  auto nnz = mat->ptr[nrows];
  for(idx_t i=0; i < nrows; ++i){
    assert(mat->ptr[i] <= nnz);
  }
  for(ptr_t j=0; j < nnz; ++j){
    assert(mat->ind[j] < ncols);
  }
}

/**
 * Given a set of collected neighbors and their associated classes, predict the class
 * of the sample who's neighbors we collected.
 * 
 * @param neighbors     Set of neighbors, including ID and similarity to the query
 * @param classes       Associated classes for the set of neighbors (array of same length as neighbors)
 * @param num_neighbors Length of neighbors and classes
 * @param sims          An array of doubles of size num_classes+1 to be used when aggregating neighbor classes.
 * @param num_classes   Maximum class ID. Classes start with 1 and are sequentially numbered.
 * 
 * @return Predicted class for the sample.
*/
int predict1(pair<idx_t, double> * neighbors, char * classes, idx_t num_neighbors, double * sims, char num_classes=5)
{
  for(int i=0; i <= num_classes; i++){
    sims[i] = 0;
  }
  for(idx_t j=0; j < num_neighbors; ++j){
    sims[(unsigned int) classes[j]] += neighbors[j].second;
  }
  double max = 0.0;
  int cls = 0;
  for(int i=1; i <= num_classes; i++){
    if(sims[i] > max){
      max = sims[i];
      cls = i;
    }
  }
  return cls;
}

/**
 * Given a set of collected neighbors and classes for all possible neighbors, predict the class
 * of the sample who's neighbors we collected.
 * 
 * @param neighbors     Set of neighbors, including ID and similarity to the query
 * @param classes       Classes for all neighbors in the DB; classes[i] is the class label for the ith DB object
 * @param num_neighbors Length of neighbors and classes
 * @param sims          An array of doubles of size num_classes+1 to be used when aggregating neighbor classes.
 * @param num_classes   Maximum class ID. Classes start with 1 and are sequentially numbered.
 * 
 * @return Predicted class for the sample.
*/
int predict2(pair<idx_t, double> * neighbors, char * classes, idx_t num_neighbors, double * sims, char num_classes=5)
{
  for(int i=1; i <= num_classes; i++){
    sims[i] = 0;
  }
  for(idx_t j=0; j < num_neighbors; ++j){
    if(neighbors[j].first == -1){ /* deal with incomplete lists with < k elements */
      break;
    }
    sims[(unsigned int) classes[neighbors[j].first]] += neighbors[j].second;
  }
  double max = 0.0;
  int cls = 0;
  for(int i=1; i <= num_classes; i++){
    if(sims[i] > max){
      max = sims[i];
      cls = i;
    }
  }
  return cls;
}



/**
 * Send the csr matrix mat to process rank.
 * @param mat CSR matrix to send
 * @param rank Process ID to send to
 * @param comm Communicator to send via
 */
void send_csr(csr_t * mat, int to, MPI_Comm comm) {
  // first send nrows and ncols
  MPI_Send(
    /* data         = */ &(mat->nrows), 
    /* count        = */ 1, 
    /* datatype     = */ MPI_UNSIGNED, 
    /* destination  = */ to, 
    /* tag          = */ 0, 
    /* communicator = */ comm);
  MPI_Send(&(mat->ncols), 1, MPI_UNSIGNED, to, 1, comm);

  // now send ptr array
  MPI_Send(mat->ptr, mat->nrows+1, MPI_UNSIGNED_LONG, to, 2, comm);
  ptr_t nnz = mat->ptr[mat->nrows];

  // now send ind
  MPI_Send(mat->ind, nnz, MPI_UNSIGNED, to, 3, comm);

  // finally send val
  MPI_Send(mat->val, nnz, MPI_FLOAT, to, 4, comm);
}

/**
 * Receive a csr matrix from process rank.
 * @param rank Process ID receiving matrix from
 * @param comm Communicator to receive via
 */
csr_t * receive_csr(int rank, MPI_Comm comm) {
  // first receive nrows and ncols
  idx_t nrows, ncols;
  MPI_Recv(
    /* data         = */ &nrows, 
    /* count        = */ 1, 
    /* datatype     = */ MPI_UNSIGNED, 
    /* source       = */ rank, 
    /* tag          = */ 0, 
    /* communicator = */ comm, 
    /* status       = */ MPI_STATUS_IGNORE);
  MPI_Recv(&ncols, 1, MPI_UNSIGNED, rank, 1, comm, MPI_STATUS_IGNORE);

  // now allocate space for ptr array and receive the ptr array
  ptr_t * ptr = (ptr_t*) malloc(sizeof(ptr_t) * (nrows+1));
  if(!ptr){
    throw std::runtime_error("Could not allocate ptr array.");
  }
  MPI_Recv(ptr, nrows+1, MPI_UNSIGNED_LONG, rank, 2, comm, MPI_STATUS_IGNORE);

  // figure out number of non-zeros
  ptr_t nnz = ptr[nrows];

  // now allocate space and receive the ind array
  idx_t * ind = (idx_t*) malloc(sizeof(idx_t) * nnz);
  if(!ind){
    throw std::runtime_error("Could not allocate ind array.");
  }
  MPI_Recv(ind, nnz, MPI_UNSIGNED, rank, 3, comm, MPI_STATUS_IGNORE);

  // now allocate space and receive the val array
  val_t * val = (val_t*) malloc(sizeof(val_t) * nnz);
  if(!val){
    throw std::runtime_error("Could not allocate val array.");
  }
  MPI_Recv(val, nnz, MPI_FLOAT, rank, 4, comm, MPI_STATUS_IGNORE);

  // now put it all together and return
  csr_t * mat = new csr_t();
  if(!mat){
    throw std::runtime_error("Could not allocate matrix structure.");
  }
  mat->nrows = nrows;
  mat->ncols = ncols;
  mat->ptr = ptr;
  mat->ind = ind;
  mat->val = val;
  return mat;
}

/** 
 * Read a subset of the rows in the CSR matrix. 
 * Stop reading at the end of the row once at least nnz non-zeros have been read.
 * Note that the file pointer may be in the middle of the matrix or right after reading the "nrows ncols nnz" line.
 * 
 * @param infile Pointer to the opened input file
 * @param rnnz   Number of non-zeros to read
 * @param nrows  Expected number of rows (may be more or less)
 * @param ncols  Number of columns in the matrix
*/
csr_t * read_partial_csr(FILE * infile, ptr_t rnnz, idx_t nrows, idx_t ncols)
{
  char * line = NULL;
  size_t n, nr, nnz;
  char * head;
  char * tail;
  idx_t cid;
  double dval;
  
  if (!infile) {
    throw std::runtime_error("Invalid file pointer\n");
  }

  //allocate space
  ptr_t mnnz = rnnz * 1.5;
  idx_t mnrows = nrows * 1.5;
  auto mat = new csr_t();
  mat->reserve(mnrows, mnnz);
  mat->ncols = ncols;
  
  //read in rowval, rowind, rowptr
  mat->ptr[0]= 0;
  nnz = 0;
  nr = 0;

  while(nnz < rnnz && getline(&line, &n, infile) != -1){
    head = line;
    while (1) {
      cid = (idx_t) strtol(head, &tail, 0);
      if (tail == head)
        break;
      head = tail;

      if(cid < 1){
        throw std::runtime_error("Invalid column ID while reading CLUTO matrix\n");
      }
      if(nnz == mnnz){ // ensure there's enough space
        mnnz *= 1.5;
        mat->reserve(mnrows, mnnz);
      }
      mat->ind[nnz] = cid - 1; //csr/clu files have 1-index based column IDs and our matrix is 0-based.
      dval = strtod(head, &tail);
      head = tail;
      mat->val[nnz++] = dval;
    }
    if (nr == mnrows){ // ensure there's enough space
      mnrows *= 1.5;
      mat->reserve(mnrows, mnnz);
    }
    mat->ptr[nr+1] = nnz;
    nr++;
  }
  // reduce the space of the matrix to what is needed
  mat->nrows = nr;
  mat->ptr = (ptr_t*) realloc(mat->ptr, sizeof(ptr_t) * (nr+1));
  mat->ind = (idx_t*) realloc(mat->ind, sizeof(idx_t) * nnz);
  mat->val = (val_t*) realloc(mat->val, sizeof(val_t) * nnz);
  free(line);
  return mat;
}

/** 
 * Read the first line in a cluto file and return nrows, ncols, and nnz.
 * 
 * @param infile Pointer to the opened input file
 * @param nrows  Pointer to where nrows should be stored
 * @param ncols  Pointer to where ncols should be stored
 * @param nnz    Pointer to where nnz should be stored
*/
void get_cluto_stats(FILE * infile, idx_t * nrows, idx_t * ncols, ptr_t * nnz)
{
  char * line = NULL;
  size_t n;
  
  if (!infile) {
    throw std::runtime_error("Invalid file pointer\n");
  }
  if(getline (&line, &n, infile) < 0){
    throw std::runtime_error("Could not read first line from CLU file\n");
  }
  //read matriz size info
  sscanf(line, "%u %u %zu", nrows, ncols, nnz);
  free(line);
}

csr_t * read_and_bcast_csr(char * fname, MPI_Comm comm)
{
  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(comm, &world_rank);
  int world_size;
  MPI_Comm_size(comm, &world_size);

  idx_t tnrows; // total number of rows across all sub-matrices
  ptr_t tnnz;   // total number of non-zeroes across all sub-matrices
  csr_t * mat;
  if (world_rank == 0) {
    // read the header of the CLUTO file
    idx_t nrows, ncols;
    ptr_t nnz;
    FILE * infile = fopen(fname, "r");
    get_cluto_stats(infile, &nrows, &ncols, &nnz);

    // read the first sub-matrix and store locally
    ptr_t pnnz = ceil(nnz/(double)world_size);
    idx_t pnrs = ceil(nrows/(double)world_size);
    mat = read_partial_csr(infile, pnnz, pnrs, ncols);

    // now send the rest of the matrices
    for(int i=1; i < world_size; ++i){
      auto smat = read_partial_csr(infile, pnnz, pnrs, ncols);
      send_csr(smat, i, comm);
      delete smat;      
    }

    // close the file
    fclose(infile);
  } else {
    mat  = receive_csr(0, comm);
  }

  return mat;
}

int main(int argc, char *argv[])
{
 
 
  if(argc < 5){
    cerr << "Invalid options." << endl << "<program> <database_file> <queries_file> <labels_file> <k> [<predfile>]" << endl;
    exit(1);
  }

  int k  = strtol(argv[4], (char **) NULL, 10);

  // Initialize the MPI environment
  MPI_Init(NULL, NULL);

  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const char * predfile = argc == 6 ? argv[5] : "predictions.txt";

  if(world_rank == 0){
    cout << "DB file: " << argv[1] << endl;
    cout << "Queries file: " << argv[2] << endl;
    cout << "Labels file: " << argv[3] << endl;
    cout << "Predictions file: " << predfile << endl;
    cout << "k: " << argv[4] << endl;
  }

  int nthreads;
  #pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }
  cout << "Process " << world_rank << ": executing using " << nthreads << " threads." << endl;

  // read and distribute the database and the queries
  if(world_rank == 0) cout << "Process 0: Broadcasting DB matrix." << endl;
  auto dbs = read_and_bcast_csr(argv[1], MPI_COMM_WORLD); // local subset of DB matrix
  cout << "Process " << world_rank << " DB chunk info: " << dbs->info() << endl;
  if(world_rank == 0) cout << "Process 0: Broadcasting Queries matrix." << endl;
  auto queries = read_and_bcast_csr(argv[2], MPI_COMM_WORLD); // local subset of Queries matrix
  cout << "Process " << world_rank << " Queries chunk info: " << queries->info() << endl;

  // normalize samples
  dbs->normalize();
  queries->normalize();

  // do your work...


  // this is just an example for getting a prediction for a given knn list
  if(world_rank == 0){
    array<pair<idx_t, double>, 6> neighbors = { { { 2, 0.3 }, { 15, 0.4 }, { 4, 0.1 }, { 11, 0.7 }, { 26, 0.2 }, { 32, 0.5 } } };
    array<char, 6> classes = {1, 5, 3, 1, 3, 4};
    double sims[6];
    cout << "Predicted class is: " << predict1(neighbors.data(), classes.data(), 6, sims) << endl;
  }

  // remember to clean up dynamically allocated memory
  delete dbs;
  delete queries;

  // Finalize the MPI environment.
  MPI_Finalize();

  return 0;
}
