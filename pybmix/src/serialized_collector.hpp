#ifndef PYBMIX_SERIALIZED_COLLECTOR_
#define PYBMIX_SERIALIZED_COLLECTOR_

#include <pybind11/pybind11.h>

#include "bayesmix/src/collectors/memory_collector.hpp"

class SerializedCollector : public BaseCollector<bayesmix::MarginalState> {
 protected:
  //! Deque that contains all states in Protobuf-object form
  std::deque<pybind11::bytes> chain;

  //! Reads the next state, based on the curr_iter curson
  bayesmix::MarginalState next_state() override {
    if (curr_iter == size - 1) {
      curr_iter = -1;
      return get_state(size - 1);
    } else {
      return get_state(curr_iter);
    }
  }

  using BaseCollector<bayesmix::MarginalState>::size;
  using BaseCollector<bayesmix::MarginalState>::curr_iter;

 public:
  // DESTRUCTOR AND CONSTRUCTORS
  ~SerializedCollector() = default;
  SerializedCollector() = default;

  //! Initializes collector (here, it does nothing)
  void start() override { return; }
  //! Closes collector (here, it does nothing)
  void finish() override { return; }

  //! Writes the given state to the collector
  void collect(bayesmix::MarginalState iter_state) override {
    std::string s;
    iter_state.SerializeToString(&s);
    chain.push_back(s);
    size++;
  }

  // GETTERS AND SETTERS
  //! Returns i-th state in the collector
  bayesmix::MarginalState get_state(unsigned int i) override {
    bayesmix::MarginalState out;
    out.ParseFromString(chain[i]);
    return out;
  }
  //! Returns the whole chain in form of a deque of States
  std::deque<bayesmix::MarginalState> get_chain() override {
    std::deque<bayesmix::MarginalState> out;
    for (int i=0; i < chain.size(); i++) {
        out.push_back(get_state(i));
    }
    return out;
   }

   pybind11::bytes get_serialized_state(unsigned int i) {
       return chain[i];
   }

   std::deque<pybind11::bytes> get_serialized_chain() {
     return chain;
   }
};

#endif