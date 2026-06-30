#include "core/layer.h"
#include "core/config.h"
#include <cstring>

// Forward declarations for per-arch creation
Layer create_gqa_layer();
Layer create_rina_ssm_layer();
Layer create_rina_mla_layer();

Layer* create_layer_by_type(const std::string& type) {
    Layer* l = new Layer();
    if (type == "standard_gqa" || type == "standard_attention" || type == "gqa") {
        *l = create_gqa_layer();
    } else if (type == "inertia_wave_ssm" || type == "ssm") {
        *l = create_rina_ssm_layer();
    } else if (type == "sparse_gather_fa" || type == "mla") {
        *l = create_rina_mla_layer();
    } else {
        delete l;
        return nullptr;
    }
    return l;
}
