#ifndef PYBMIX_LOAD_MIXINGS_2_H
#define PYBMIX_LOAD_MIXINGS_2_H

#include <functional>
#include <memory>

#include "bayesmix/src/mixings/abstract_mixing.h"
#include "bayesmix/src/runtime/factory.h"
#include "python_mixing.h"
#include <iostream>

//! Loads all available `Mixing` objects into the appropriate factory, so that
//! they are ready to be chosen and used at runtime.

template <class AbstractProduct>
using Builder = std::function<std::shared_ptr<AbstractProduct>()>;

using MixingFactory = Factory<bayesmix::MixingId, AbstractMixing>;

__attribute__((constructor)) static void load_py_mixings() {
    MixingFactory &factory = MixingFactory::Instance();
    // Initialize factory builders
    Builder<AbstractMixing> Pythonbuilder = []() {
        return std::make_shared<PythonMixing>();
    };
    factory.add_builder(PythonMixing().get_id(), Pythonbuilder);
}

#endif //PYBMIX_LOAD_MIXINGS_2_H
