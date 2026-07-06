#pragma once
#include <string>
#include <vector>
#include <functional>
#include "core/config.h"
#include "core/quant.h"
#include "loader/gguf_reader.h"

// ── Tensor name mapping ──────────────────────────────────────
// GGUF name → RINN internal name for a single tensor
struct TensorNameEntry {
    const char* gguf_name;
    const char* rinn_name;
};

// ── Architecture-specific loader interface ───────────────────
struct ArchLoader {
    std::string name;

    // Detect if this architecture matches the GGUF metadata
    bool (*detect)(const GGUFMetadata& meta) = nullptr;

    // Map GGUF tensor names to RINN internal names.
    // Returns false if any required tensor is missing.
    // fill from const TensorNameEntry table
    bool (*map_names)(const GGUFMetadata& meta,
                      std::vector<TensorNameEntry>& out_names) = nullptr;

    // Build ModelConfig from GGUF metadata
    bool (*build_config)(ModelConfig& cfg, const GGUFMetadata& meta) = nullptr;
};

// ── Architecture loader registry ─────────────────────────────
void register_loader(const ArchLoader& loader);
const std::vector<ArchLoader>& get_registered_loaders();
const ArchLoader* detect_arch(const GGUFMetadata& meta);

// ── Shared loader utilities ──────────────────────────────────
bool should_quant_weight(const std::string& rinn_name);
void generate_rope_tables(int max_seq, int head_dim,
                          std::vector<float>& cos_tbl,
                          std::vector<float>& sin_tbl,
                          float freq_base = 10000.0f,
                          float yarn_scale = 1.0f);

// Convert GGML quant type to our internal QuantType
QuantType ggml_to_quant_type(int ggml_type);

// ── Single buffer allocation ────────────────────────────────
// Allocate one big GPU buffer for all weights, sub-allocate by offset.
// Returns the base pointer, sets total_bytes to the allocated size.
// The size includes padding for alignment.
void* allocate_weight_buffer(const std::vector<GGUFTensorInfo>& tensors,
                             size_t& total_bytes);

// Offset of a tensor within the single weight buffer (cumulative)
size_t tensor_buffer_offset(const std::vector<GGUFTensorInfo>& tensors, int idx);
