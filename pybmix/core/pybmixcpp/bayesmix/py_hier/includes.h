#ifndef BAYESMIX_INCLUDES_2_H_
#define BAYESMIX_INCLUDES_2_H_

#include "algorithm_params.pb.h"
#include "bayesmix/src/algorithms/blocked_gibbs_algorithm.h"
#include "bayesmix/src/algorithms/load_algorithms.h"
#include "bayesmix/src/algorithms/neal2_algorithm.h"
#include "bayesmix/src/algorithms/neal3_algorithm.h"
#include "bayesmix/src/algorithms/neal8_algorithm.h"
#include "bayesmix/src/collectors/file_collector.h"
#include "bayesmix/src/collectors/memory_collector.h"
#include "bayesmix/src/hierarchies/fa_hierarchy.h"
#include "bayesmix/src/hierarchies/lapnig_hierarchy.h"
#include "bayesmix/src/hierarchies/lin_reg_uni_hierarchy.h"
#include "bayesmix/src/hierarchies/nnig_hierarchy.h"
#include "bayesmix/src/hierarchies/nnw_hierarchy.h"
#include "bayesmix/src/mixings/dirichlet_mixing.h"
#include "bayesmix/src/mixings/load_mixings.h"
#include "bayesmix/src/mixings/logit_sb_mixing.h"
#include "bayesmix/src/mixings/mixture_finite_mixing.h"
#include "bayesmix/src/mixings/pityor_mixing.h"
#include "bayesmix/src/mixings/truncated_sb_mixing.h"
#include "bayesmix/src/runtime/factory.h"
#include "bayesmix/src/utils/cluster_utils.h"
#include "bayesmix/src/utils/eval_like.h"
#include "bayesmix/src/utils/io_utils.h"
#include "bayesmix/src/utils/proto_utils.h"
#include "load_hierarchies_2.h"

#endif  // BAYESMIX_INCLUDES_2_H_