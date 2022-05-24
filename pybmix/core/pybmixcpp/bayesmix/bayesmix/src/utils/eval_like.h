#ifndef BAYESMIX_UTILS_EVAL_LIKE_H_
#define BAYESMIX_UTILS_EVAL_LIKE_H_

#include "src/includes.h"
#include "src/utils/eigen_utils.h"

namespace bayesmix {

//! * Evaluates the (mixture) likelihood for all the states of the MCMC chain,
//! in parallel.
//! @param algo a shared_ptr to the algorithm used for MCMC sampling
//! @param collector a pointer to the collector containing the chain
//! @param low_memory if false, the whole chain will be loaded into the memory.
//!        This leads to a speedup which ranges between 20-50% but increases
//!        significantly the memory required.
//!        If true, we load the chain by deserializing chunks of size
//!        `chunk_size`, and process the states within each chunk in parallel.
//! @param njobs used only if low_memory=false, the number of parallel jobs
//! @param chunk_size see `low_memory`.
Eigen::MatrixXd eval_lpdf_parallel(
    const std::shared_ptr<BaseAlgorithm> algo, BaseCollector *const collector,
    const Eigen::MatrixXd &grid,
    const Eigen::RowVectorXd &hier_covariate = Eigen::RowVectorXd(0),
    const Eigen::RowVectorXd &mix_covariate = Eigen::RowVectorXd(0),
    const bool low_memory = false, const int njobs = 4,
    const int chunk_size = 100);

namespace internal {

template <typename T>
std::vector<std::vector<T>> gen_even_slices(const std::vector<T> &x,
                                            const int num_slices) {
  std::vector<std::vector<T>> out(num_slices);
  for (int i = 0; i < x.size(); i++) {
    out[i % num_slices].push_back(x[i]);
  }
  return out;
}

Eigen::MatrixXd eval_lpdf_parallel_lowmemory(
    const std::shared_ptr<BaseAlgorithm> algo, BaseCollector *const collector,
    const Eigen::MatrixXd &grid,
    const Eigen::RowVectorXd &hier_covariate = Eigen::RowVectorXd(0),
    const Eigen::RowVectorXd &mix_covariate = Eigen::RowVectorXd(0),
    const int chunk_size = 100);

Eigen::MatrixXd eval_lpdf_parallel_fullmemory(
    const std::shared_ptr<BaseAlgorithm> algo, BaseCollector *const collector,
    const Eigen::MatrixXd &grid,
    const Eigen::RowVectorXd &hier_covariate = Eigen::RowVectorXd(0),
    const Eigen::RowVectorXd &mix_covariate = Eigen::RowVectorXd(0),
    const int njobs = 4);
}  // namespace internal

}  // namespace bayesmix

#endif
