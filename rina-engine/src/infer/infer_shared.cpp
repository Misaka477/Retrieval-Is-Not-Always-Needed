#include "infer/infer_base.h"
#include <unordered_map>

// Forward declarations for factory functions
Inference* create_gqa_inference();
Inference* create_mla_inference();

static std::unordered_map<std::string, InferenceFactory> s_factories;

void register_inference(const std::string& name, InferenceFactory factory) {
    s_factories[name] = factory;
}

Inference* create_inference(const std::string& name) {
    auto it = s_factories.find(name);
    if (it != s_factories.end()) return it->second();
    return nullptr;
}

void register_all_inferences() {
    register_inference("gqa", create_gqa_inference);
    register_inference("mla", create_mla_inference);
}
