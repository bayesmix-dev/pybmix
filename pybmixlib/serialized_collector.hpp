#ifndef PYBMIX_SERIALIZED_COLLECTOR_
#define PYBMIX_SERIALIZED_COLLECTOR_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bayesmix/src/collectors/memory_collector.h"

class SerializedCollector : public MemoryCollector {
 public:
  ~SerializedCollector() = default;
  SerializedCollector() = default;

  pybind11::bytes get_serialized_state(unsigned int i) const {
    return (pybind11::bytes)chain[i];
  }

  std::vector<pybind11::bytes> get_serialized_chain() const {
    std::vector<pybind11::bytes> out(chain.size());
    for (int i = 0; i < chain.size(); i++) out[i] = get_serialized_state(i);

    return out;
  }
};

void add_serialized_collector(pybind11::module &m);

#endif
