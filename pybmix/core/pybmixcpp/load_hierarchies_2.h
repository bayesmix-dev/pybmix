#ifndef BAYESMIX_HIERARCHIES_LOAD_HIERARCHIES_2_H_
#define BAYESMIX_HIERARCHIES_LOAD_HIERARCHIES_2_H_

#include <functional>
#include <memory>

#include "bayesmix/src/hierarchies/abstract_hierarchy.h"
#include "python_hierarchy.h"
#include "bayesmix/src/runtime/factory.h"

//! Loads all available `Hierarchy` objects into the appropriate factory, so
//! that they are ready to be chosen and used at runtime.

template <class AbstractProduct>
using Builder = std::function<std::shared_ptr<AbstractProduct>()>;

using HierarchyFactory = Factory<bayesmix::HierarchyId, AbstractHierarchy>;

__attribute__((constructor)) static void load_hierarchies_2() {
    HierarchyFactory &factory = HierarchyFactory::Instance();

    Builder<AbstractHierarchy> Pythonbuilder = []() {
        return std::make_shared<PythonHierarchy>();
    };

    factory.add_builder(PythonHierarchy().get_id(), Pythonbuilder);
}

#endif  // BAYESMIX_HIERARCHIES_LOAD_HIERARCHIES_2_H_
