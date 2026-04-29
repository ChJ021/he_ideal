#include "openfhe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace lbcrypto;
namespace fs = std::filesystem;

using Ctxt = Ciphertext<DCRTPoly>;

struct CostCounts {
    double latency_ms = 0.0;
    int rotations = 0;
    int ct_ct_mults = 0;
    int ct_pt_mults = 0;
    int rescale_count = 0;
    int relin_count = 0;
    int depth = 0;
    int bootstrap_count = 0;
    double memory_mb = 0.0;
};

struct Request {
    std::string case_name;
    std::string schedule_name;
    std::string runner_mode = "openfhe_schedule_workload";
    fs::path schedule_path;
    fs::path output_dir;
    fs::path forward_manifest_path;
    int sample_size = 0;
    int sequence_length = 0;
    int latency_repetitions = 1;
    int multiplicative_depth = 60;
    int scaling_mod_size = 59;
    int first_mod_size = 60;
    int poly_modulus_degree = 0;
    std::string linear_kernel = "bsgs_hoisted";
    int bsgs_baby_step = 32;
    bool fuse_qkv = true;
    std::string packing_strategy = "row_packed";
    std::string token_block_size = "auto";
    bool profile_native_stages = true;
    bool bootstrap_enabled = false;
    std::vector<uint32_t> bootstrap_level_budget = {4, 4};
    std::vector<uint32_t> bootstrap_dim1 = {0, 0};
    int bootstrap_levels_after = 10;
    int bootstrap_num_iterations = 1;
    int bootstrap_precision = 0;
    bool bootstrap_auto_guard = true;
    int bootstrap_guard_min_levels = 2;
};

struct Blob {
    std::string name;
    std::string dtype;
    std::vector<std::size_t> shape;
    fs::path path;
    std::vector<double> f64;
    std::vector<int> i32;
};

struct GeluApprox {
    int layer_index = 0;
    std::string operator_name;
    std::string candidate_id;
    int degree = 0;
    double scale = 0.0;
    std::vector<double> power_coefficients;
};

struct Manifest {
    fs::path path;
    fs::path base_dir;
    std::string case_name;
    std::string schedule_name;
    int sample_count = 0;
    int sequence_length = 0;
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_layers = 0;
    int num_heads = 1;
    int num_labels = 2;
    std::unordered_map<std::string, Blob> blobs;
    std::map<std::string, std::string> candidates;
    std::map<std::string, std::string> bootstrap_policies;
    std::map<std::string, GeluApprox> gelu_approximations;
    GeluApprox default_gelu;
    bool has_default_gelu = false;
};

struct EncVec {
    Ctxt ct;
    std::size_t logical_size = 0;
};

struct StageTimes {
    double keygen_ms = 0.0;
    double encrypt_ms = 0.0;
    double qkv_ms = 0.0;
    double attention_ms = 0.0;
    double ffn_ms = 0.0;
    double classifier_ms = 0.0;
    double decrypt_ms = 0.0;
    double bootstrap_ms = 0.0;
    int rotation_key_count = 0;
    int bootstrap_policy_count = 0;
    int bootstrap_auto_count = 0;
    int plaintext_cache_entries = 0;
};

double elapsed_ms(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

std::string read_file(const fs::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to read file: " + path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void write_file(const fs::path& path, const std::string& text) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to write file: " + path.string());
    }
    output << text;
}

std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    for (const char ch : value) {
        if (ch == '\\' || ch == '"') {
            escaped << '\\' << ch;
        }
        else if (ch == '\n') {
            escaped << "\\n";
        }
        else {
            escaped << ch;
        }
    }
    return escaped.str();
}

std::string json_string(const std::string& json, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        return "";
    }
    return match[1].str();
}

int json_int(const std::string& json, const std::string& key, int fallback) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?[0-9]+)");
    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        return fallback;
    }
    return std::stoi(match[1].str());
}

bool json_bool(const std::string& json, const std::string& key, bool fallback) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(true|false)");
    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        return fallback;
    }
    return match[1].str() == "true";
}

void trim(std::string& value) {
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), value.end());
}

std::vector<uint32_t> json_uint_array(
    const std::string& json,
    const std::string& key,
    const std::vector<uint32_t>& fallback) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        return fallback;
    }
    std::vector<uint32_t> values;
    std::stringstream stream(match[1].str());
    std::string item;
    while (std::getline(stream, item, ',')) {
        trim(item);
        if (!item.empty()) {
            values.push_back(static_cast<uint32_t>(std::stoul(item)));
        }
    }
    return values.empty() ? fallback : values;
}

std::vector<uint32_t> normalize_pair(std::vector<uint32_t> values, const std::vector<uint32_t>& fallback) {
    if (values.size() < 2) {
        values = fallback;
    }
    values.resize(2);
    return values;
}

Request load_request(const fs::path& path) {
    const std::string json = read_file(path);
    Request request;
    request.case_name = json_string(json, "case_name");
    request.schedule_name = json_string(json, "schedule_name");
    request.runner_mode = json_string(json, "runner_mode");
    if (request.runner_mode.empty()) {
        request.runner_mode = "openfhe_schedule_workload";
    }
    request.schedule_path = json_string(json, "schedule_path");
    request.output_dir = json_string(json, "output_dir");
    request.forward_manifest_path = json_string(json, "forward_manifest_path");
    request.sample_size = json_int(json, "sample_size", 0);
    request.sequence_length = json_int(json, "sequence_length", 0);
    request.latency_repetitions = std::max(1, json_int(json, "latency_repetitions", 1));
    request.multiplicative_depth = json_int(json, "multiplicative_depth", 60);
    request.scaling_mod_size = json_int(json, "scaling_mod_size", 59);
    request.first_mod_size = json_int(json, "first_mod_size", 60);
    request.poly_modulus_degree = json_int(json, "poly_modulus_degree", 0);
    request.linear_kernel = json_string(json, "linear_kernel");
    if (request.linear_kernel.empty()) {
        request.linear_kernel = "bsgs_hoisted";
    }
    request.bsgs_baby_step = std::max(1, json_int(json, "bsgs_baby_step", 32));
    request.fuse_qkv = json_bool(json, "fuse_qkv", true);
    request.packing_strategy = json_string(json, "packing_strategy");
    if (request.packing_strategy.empty()) {
        request.packing_strategy = "row_packed";
    }
    request.token_block_size = json_string(json, "token_block_size");
    if (request.token_block_size.empty()) {
        request.token_block_size = "auto";
    }
    request.profile_native_stages = json_bool(json, "profile_native_stages", true);
    request.bootstrap_enabled = json_bool(json, "bootstrap_enabled", false);
    request.bootstrap_level_budget = normalize_pair(
        json_uint_array(json, "bootstrap_level_budget", {4, 4}),
        {4, 4});
    request.bootstrap_dim1 = normalize_pair(
        json_uint_array(json, "bootstrap_dim1", {0, 0}),
        {0, 0});
    request.bootstrap_levels_after = std::max(1, json_int(json, "bootstrap_levels_after", 10));
    request.bootstrap_num_iterations = std::max(1, json_int(json, "bootstrap_num_iterations", 1));
    request.bootstrap_precision = std::max(0, json_int(json, "bootstrap_precision", 0));
    request.bootstrap_auto_guard = json_bool(json, "bootstrap_auto_guard", true);
    request.bootstrap_guard_min_levels = std::max(0, json_int(json, "bootstrap_guard_min_levels", 2));
    if (request.case_name.empty() || request.schedule_name.empty() || request.output_dir.empty()) {
        throw std::runtime_error("request JSON is missing required deployment fields");
    }
    if (request.runner_mode == "openfhe_schedule_workload" && request.schedule_path.empty()) {
        throw std::runtime_error("schedule workload mode requires schedule_path");
    }
    if (request.runner_mode == "openfhe_distilbert_forward" && request.forward_manifest_path.empty()) {
        throw std::runtime_error("distilbert forward mode requires forward_manifest_path");
    }
    return request;
}

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    std::istringstream stream(line);
    while (std::getline(stream, current, ',')) {
        fields.push_back(current);
    }
    return fields;
}

int parse_int_field(const std::map<std::string, std::string>& row, const std::string& key) {
    const auto it = row.find(key);
    if (it == row.end() || it->second.empty()) {
        return 0;
    }
    return static_cast<int>(std::stod(it->second));
}

double parse_double_field(const std::map<std::string, std::string>& row, const std::string& key) {
    const auto it = row.find(key);
    if (it == row.end() || it->second.empty()) {
        return 0.0;
    }
    return std::stod(it->second);
}

bool load_counts_from_metrics(const fs::path& output_dir, const std::string& schedule_name, CostCounts& counts) {
    const fs::path run_dir = output_dir.parent_path().parent_path();
    const fs::path metrics_path = run_dir / "he_analysis" / "he_metrics.csv";
    std::ifstream input(metrics_path);
    if (!input) {
        return false;
    }

    std::string header_line;
    if (!std::getline(input, header_line)) {
        return false;
    }
    const auto headers = split_csv(header_line);
    std::string line;
    while (std::getline(input, line)) {
        const auto fields = split_csv(line);
        std::map<std::string, std::string> row;
        for (std::size_t i = 0; i < headers.size() && i < fields.size(); ++i) {
            row[headers[i]] = fields[i];
        }
        if (row["schedule"] != schedule_name) {
            continue;
        }
        counts.rotations = parse_int_field(row, "rotations");
        counts.ct_ct_mults = parse_int_field(row, "ct_ct_mults");
        counts.ct_pt_mults = parse_int_field(row, "ct_pt_mults");
        counts.rescale_count = parse_int_field(row, "rescale_count");
        counts.relin_count = parse_int_field(row, "relin_count");
        counts.depth = parse_int_field(row, "depth");
        counts.bootstrap_count = parse_int_field(row, "bootstrap_count");
        counts.memory_mb = parse_double_field(row, "memory_mb");
        return true;
    }
    return false;
}

CostCounts candidate_cost(const std::string& candidate_id) {
    if (candidate_id.find(".base") != std::string::npos) {
        return {};
    }
    if (candidate_id == "gelu.exact.high.v1") {
        return {.rotations = 0, .ct_ct_mults = 6, .ct_pt_mults = 7, .rescale_count = 5, .relin_count = 1, .depth = 5};
    }
    if (candidate_id == "gelu.chebyshev.degree11.v1") {
        return {.rotations = 0, .ct_ct_mults = 5, .ct_pt_mults = 6, .rescale_count = 5, .relin_count = 1, .depth = 5};
    }
    if (candidate_id == "gelu.chebyshev.degree9.v1") {
        return {.rotations = 0, .ct_ct_mults = 4, .ct_pt_mults = 5, .rescale_count = 4, .relin_count = 1, .depth = 4};
    }
    if (candidate_id == "gelu.chebyshev.degree5.v1") {
        return {.rotations = 0, .ct_ct_mults = 3, .ct_pt_mults = 4, .rescale_count = 3, .relin_count = 1, .depth = 3};
    }
    if (candidate_id == "layernorm.exact.high.v1") {
        return {.rotations = 2, .ct_ct_mults = 6, .ct_pt_mults = 7, .rescale_count = 6, .relin_count = 1, .depth = 6};
    }
    if (candidate_id == "layernorm.newton.low_iter.v1") {
        return {.rotations = 2, .ct_ct_mults = 3, .ct_pt_mults = 0, .rescale_count = 3, .relin_count = 1, .depth = 3};
    }
    if (candidate_id == "layernorm.centered.mid_cost.v1") {
        return {.rotations = 1, .ct_ct_mults = 1, .ct_pt_mults = 2, .rescale_count = 1, .relin_count = 1, .depth = 1};
    }
    if (candidate_id == "layernorm.affine.low_cost.v1") {
        return {.rotations = 0, .ct_ct_mults = 0, .ct_pt_mults = 1, .rescale_count = 0, .relin_count = 0, .depth = 0};
    }
    if (candidate_id == "softmax.exact.high.v1") {
        return {.rotations = 3, .ct_ct_mults = 4, .ct_pt_mults = 4, .rescale_count = 4, .relin_count = 1, .depth = 4};
    }
    if (candidate_id == "softmax.poly_exp.degree2.v1") {
        return {.rotations = 1, .ct_ct_mults = 2, .ct_pt_mults = 2, .rescale_count = 2, .relin_count = 1, .depth = 2};
    }
    if (candidate_id == "softmax.power.degree2.v1") {
        return {.rotations = 1, .ct_ct_mults = 1, .ct_pt_mults = 2, .rescale_count = 1, .relin_count = 1, .depth = 1};
    }
    return {};
}

void add_counts(CostCounts& total, const CostCounts& delta) {
    total.rotations += delta.rotations;
    total.ct_ct_mults += delta.ct_ct_mults;
    total.ct_pt_mults += delta.ct_pt_mults;
    total.rescale_count += delta.rescale_count;
    total.relin_count += delta.relin_count;
    total.depth += delta.depth;
    total.bootstrap_count += delta.bootstrap_count;
    total.memory_mb += delta.memory_mb;
}

CostCounts load_counts_from_schedule(const fs::path& schedule_path) {
    const std::string yaml = read_file(schedule_path);
    CostCounts counts;
    const std::regex candidate_pattern("candidate_id:\\s*([^\\s]+)");
    for (auto it = std::sregex_iterator(yaml.begin(), yaml.end(), candidate_pattern);
         it != std::sregex_iterator(); ++it) {
        std::string candidate_id = (*it)[1].str();
        candidate_id.erase(std::remove(candidate_id.begin(), candidate_id.end(), '"'), candidate_id.end());
        candidate_id.erase(std::remove(candidate_id.begin(), candidate_id.end(), '\''), candidate_id.end());
        add_counts(counts, candidate_cost(candidate_id));
    }

    const std::regex bootstrap_pattern("bootstrap_policy:\\s*bootstrap_before");
    counts.bootstrap_count = static_cast<int>(
        std::distance(std::sregex_iterator(yaml.begin(), yaml.end(), bootstrap_pattern), std::sregex_iterator()));
    return counts;
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double index = q * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(index));
    const auto upper = static_cast<std::size_t>(std::ceil(index));
    if (lower == upper) {
        return values[lower];
    }
    const double weight = index - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

double run_openfhe_workload_once(const CostCounts& counts) {
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(2);
    parameters.SetScalingModSize(48);
    parameters.SetBatchSize(8);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    if (counts.rotations > 0) {
        cc->EvalAtIndexKeyGen(keys.secretKey, {1});
    }

    std::vector<std::complex<double>> values(8);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = std::complex<double>(1.0 + 0.01 * static_cast<double>(i), 0.0);
    }
    auto plaintext = cc->MakeCKKSPackedPlaintext(values);
    auto ciphertext = cc->Encrypt(keys.publicKey, plaintext);
    auto last = ciphertext;

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < counts.ct_ct_mults; ++i) {
        last = cc->EvalMult(ciphertext, ciphertext);
    }
    for (int i = 0; i < counts.ct_pt_mults; ++i) {
        last = cc->EvalMult(ciphertext, plaintext);
    }
    for (int i = 0; i < counts.rotations; ++i) {
        last = cc->EvalAtIndex(ciphertext, 1);
    }
    const auto stop = std::chrono::steady_clock::now();

    Plaintext decrypted;
    cc->Decrypt(keys.secretKey, last, &decrypted);
    decrypted->SetLength(1);
    if (decrypted->GetCKKSPackedValue().empty()) {
        throw std::runtime_error("OpenFHE workload produced empty decrypt result");
    }
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

std::string workload_result_json(const Request& request, const CostCounts& counts, const std::vector<double>& latencies) {
    const double mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) /
                        static_cast<double>(std::max<std::size_t>(latencies.size(), 1));
    const double p50 = percentile(latencies, 0.50);
    const double p95 = percentile(latencies, 0.95);

    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"case_name\": \"" << json_escape(request.case_name) << "\",\n";
    out << "  \"schedule_name\": \"" << json_escape(request.schedule_name) << "\",\n";
    out << "  \"feasible\": true,\n";
    out << "  \"accuracy\": null,\n";
    out << "  \"sample_count\": " << request.sample_size << ",\n";
    out << "  \"latency_ms\": " << mean << ",\n";
    out << "  \"latency_p50_ms\": " << p50 << ",\n";
    out << "  \"latency_p95_ms\": " << p95 << ",\n";
    out << "  \"predictions_path\": \"\",\n";
    out << "  \"error\": \"\",\n";
    out << "  \"cost\": {\n";
    out << "    \"latency_ms\": " << mean << ",\n";
    out << "    \"rotations\": " << counts.rotations << ",\n";
    out << "    \"ct_ct_mults\": " << counts.ct_ct_mults << ",\n";
    out << "    \"ct_pt_mults\": " << counts.ct_pt_mults << ",\n";
    out << "    \"rescale_count\": " << counts.rescale_count << ",\n";
    out << "    \"relin_count\": " << counts.relin_count << ",\n";
    out << "    \"depth\": " << counts.depth << ",\n";
    out << "    \"bootstrap_count\": " << counts.bootstrap_count << ",\n";
    out << "    \"memory_mb\": " << counts.memory_mb << "\n";
    out << "  },\n";
    out << "  \"backend_metadata\": {\n";
    out << "    \"runner\": \"hetune_openfhe_runner\",\n";
    out << "    \"runner_mode\": \"openfhe_schedule_workload\",\n";
    out << "    \"openfhe_workload\": \"ckks_ciphertext_ops_from_schedule_costs\",\n";
    out << "    \"bootstrap_mode\": \"counted_not_executed\",\n";
    out << "    \"accuracy_source\": \"plaintext_evaluation_metrics_if_available\"\n";
    out << "  }\n";
    out << "}\n";
    return out.str();
}

std::vector<std::size_t> parse_shape(const std::string& value) {
    std::vector<std::size_t> shape;
    const std::regex number("([0-9]+)");
    for (auto it = std::sregex_iterator(value.begin(), value.end(), number);
         it != std::sregex_iterator(); ++it) {
        shape.push_back(static_cast<std::size_t>(std::stoul((*it)[1].str())));
    }
    return shape;
}

std::vector<double> parse_double_array(const std::string& value) {
    std::vector<double> out;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        trim(item);
        if (!item.empty()) {
            out.push_back(std::stod(item));
        }
    }
    return out;
}

std::size_t shape_size(const std::vector<std::size_t>& shape) {
    if (shape.empty()) {
        return 0;
    }
    return std::accumulate(shape.begin(), shape.end(), static_cast<std::size_t>(1), std::multiplies<std::size_t>());
}

std::vector<double> read_float32_file(const fs::path& path, std::size_t count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to read float32 blob: " + path.string());
    }
    std::vector<float> raw(count);
    input.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (static_cast<std::size_t>(input.gcount()) != count * sizeof(float)) {
        throw std::runtime_error("short float32 blob: " + path.string());
    }
    std::vector<double> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = static_cast<double>(raw[i]);
    }
    return values;
}

std::vector<int> read_int32_file(const fs::path& path, std::size_t count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to read int32 blob: " + path.string());
    }
    std::vector<int32_t> raw(count);
    input.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(count * sizeof(int32_t)));
    if (static_cast<std::size_t>(input.gcount()) != count * sizeof(int32_t)) {
        throw std::runtime_error("short int32 blob: " + path.string());
    }
    return std::vector<int>(raw.begin(), raw.end());
}

Manifest load_manifest(const fs::path& path) {
    const std::string json = read_file(path);
    Manifest manifest;
    manifest.path = path;
    manifest.base_dir = path.parent_path();
    manifest.case_name = json_string(json, "case_name");
    manifest.schedule_name = json_string(json, "schedule_name");
    manifest.sample_count = json_int(json, "sample_count", 0);
    manifest.sequence_length = json_int(json, "sequence_length", 0);
    manifest.hidden_size = json_int(json, "hidden_size", 0);
    manifest.intermediate_size = json_int(json, "intermediate_size", 0);
    manifest.num_layers = json_int(json, "num_layers", 0);
    manifest.num_heads = std::max(1, json_int(json, "num_heads", 1));
    manifest.num_labels = std::max(1, json_int(json, "num_labels", 2));

    const std::regex blob_pattern(
        "\\{\\s*\"name\"\\s*:\\s*\"([^\"]+)\"\\s*,\\s*\"dtype\"\\s*:\\s*\"([^\"]+)\"\\s*,\\s*\"shape\"\\s*:\\s*\\[([^\\]]*)\\]\\s*,\\s*\"path\"\\s*:\\s*\"([^\"]+)\"\\s*\\}");
    for (auto it = std::sregex_iterator(json.begin(), json.end(), blob_pattern);
         it != std::sregex_iterator(); ++it) {
        Blob blob;
        blob.name = (*it)[1].str();
        blob.dtype = (*it)[2].str();
        blob.shape = parse_shape((*it)[3].str());
        blob.path = manifest.base_dir / (*it)[4].str();
        const auto count = shape_size(blob.shape);
        if (blob.dtype == "float32") {
            blob.f64 = read_float32_file(blob.path, count);
        }
        else if (blob.dtype == "int32") {
            blob.i32 = read_int32_file(blob.path, count);
        }
        else {
            throw std::runtime_error("unsupported blob dtype for " + blob.name + ": " + blob.dtype);
        }
        manifest.blobs[blob.name] = std::move(blob);
    }

    const std::regex schedule_pattern(
        "\\{\\s*\"layer_index\"\\s*:\\s*([0-9]+)\\s*,\\s*\"operator_type\"\\s*:\\s*\"([^\"]+)\"\\s*,\\s*\"operator_name\"\\s*:\\s*\"([^\"]+)\"[\\s\\S]*?\"candidate_id\"\\s*:\\s*\"([^\"]+)\"[\\s\\S]*?\"bootstrap_policy\"\\s*:\\s*\"([^\"]*)\"");
    for (auto it = std::sregex_iterator(json.begin(), json.end(), schedule_pattern);
         it != std::sregex_iterator(); ++it) {
        const int layer = std::stoi((*it)[1].str());
        const std::string op = (*it)[2].str();
        const std::string name = (*it)[3].str();
        const std::string candidate = (*it)[4].str();
        const std::string bootstrap_policy = (*it)[5].str();
        const std::string exact_key = std::to_string(layer) + "|" + op + "|" + name;
        manifest.candidates[exact_key] = candidate;
        manifest.candidates[std::to_string(layer) + "|" + op + "|"] = candidate;
        manifest.bootstrap_policies[exact_key] = bootstrap_policy;
    }

    const std::regex gelu_pattern(
        "\\{\\s*\"layer_index\"\\s*:\\s*(-?[0-9]+)\\s*,\\s*\"operator_name\"\\s*:\\s*\"([^\"]+)\"[\\s\\S]*?\"candidate_id\"\\s*:\\s*\"([^\"]+)\"[\\s\\S]*?\"degree\"\\s*:\\s*([0-9]+)\\s*,\\s*\"scale\"\\s*:\\s*([-+0-9.eE]+)[\\s\\S]*?\"power_coefficients\"\\s*:\\s*\\[([^\\]]*)\\]");
    for (auto it = std::sregex_iterator(json.begin(), json.end(), gelu_pattern);
         it != std::sregex_iterator(); ++it) {
        GeluApprox approx;
        approx.layer_index = std::stoi((*it)[1].str());
        approx.operator_name = (*it)[2].str();
        approx.candidate_id = (*it)[3].str();
        approx.degree = std::stoi((*it)[4].str());
        approx.scale = std::stod((*it)[5].str());
        approx.power_coefficients = parse_double_array((*it)[6].str());
        if (approx.power_coefficients.empty()) {
            throw std::runtime_error("GELU approximation has no power coefficients: " + approx.candidate_id);
        }
        if (approx.layer_index < 0) {
            manifest.default_gelu = approx;
            manifest.has_default_gelu = true;
        }
        else {
            manifest.gelu_approximations[std::to_string(approx.layer_index) + "|" + approx.operator_name] = approx;
        }
    }

    if (manifest.sample_count <= 0 || manifest.sequence_length <= 0 || manifest.hidden_size <= 0 ||
        manifest.num_layers <= 0) {
        throw std::runtime_error("forward manifest is missing required model dimensions");
    }
    return manifest;
}

const Blob& blob(const Manifest& manifest, const std::string& name) {
    const auto it = manifest.blobs.find(name);
    if (it == manifest.blobs.end()) {
        throw std::runtime_error("manifest missing blob: " + name);
    }
    return it->second;
}

std::size_t next_power_of_two(std::size_t value) {
    std::size_t out = 1;
    while (out < value) {
        out <<= 1U;
    }
    return out;
}

std::vector<double> padded(const std::vector<double>& values, std::size_t slots) {
    std::vector<double> out(slots, 0.0);
    const std::size_t limit = std::min(values.size(), slots);
    std::copy(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(limit), out.begin());
    return out;
}

std::vector<int32_t> required_rotation_indices(const Request& request, std::size_t slots) {
    std::set<int32_t> indices;
    if (request.linear_kernel == "diagonal") {
        for (std::size_t i = 1; i < slots; ++i) {
            indices.insert(static_cast<int32_t>(i));
        }
    }
    else {
        const std::size_t baby_step = std::min<std::size_t>(
            std::max<std::size_t>(1, static_cast<std::size_t>(request.bsgs_baby_step)),
            slots);
        for (std::size_t i = 1; i < baby_step; ++i) {
            indices.insert(static_cast<int32_t>(i));
        }
        for (std::size_t i = baby_step; i < slots; i += baby_step) {
            indices.insert(static_cast<int32_t>(i));
        }
    }
    for (std::size_t shift = 1; shift < slots; shift <<= 1U) {
        indices.insert(static_cast<int32_t>(shift));
    }
    return std::vector<int32_t>(indices.begin(), indices.end());
}

class HEEngine {
  public:
    HEEngine(const Request& request, std::size_t slots, CostCounts& counts)
        : request_(request), slots_(slots), counts_(counts) {
        CCParams<CryptoContextCKKSRNS> parameters;
        const SecretKeyDist secret_key_dist = UNIFORM_TERNARY;
        parameters.SetSecretKeyDist(secret_key_dist);
        actual_multiplicative_depth_ = std::max(2, request.multiplicative_depth);
        if (request.bootstrap_enabled) {
            const uint32_t bootstrap_depth =
                FHECKKSRNS::GetBootstrapDepth(request.bootstrap_level_budget, secret_key_dist);
            actual_multiplicative_depth_ = std::max(
                actual_multiplicative_depth_,
                static_cast<int>(request.bootstrap_levels_after + bootstrap_depth));
        }
        parameters.SetMultiplicativeDepth(static_cast<uint32_t>(actual_multiplicative_depth_));
        parameters.SetScalingModSize(static_cast<uint32_t>(std::max(30, request.scaling_mod_size)));
        parameters.SetFirstModSize(static_cast<uint32_t>(std::max(40, request.first_mod_size)));
        parameters.SetBatchSize(static_cast<uint32_t>(slots_));
        parameters.SetScalingTechnique(FLEXIBLEAUTO);
        if (request.poly_modulus_degree > 0) {
            parameters.SetSecurityLevel(HEStd_NotSet);
            parameters.SetRingDim(static_cast<uint32_t>(request.poly_modulus_degree));
        }
        cc_ = GenCryptoContext(parameters);
        cc_->Enable(PKE);
        cc_->Enable(KEYSWITCH);
        cc_->Enable(LEVELEDSHE);
        if (request.bootstrap_enabled) {
            cc_->Enable(ADVANCEDSHE);
            cc_->Enable(FHE);
            cc_->EvalBootstrapSetup(request.bootstrap_level_budget, request.bootstrap_dim1, static_cast<uint32_t>(slots_));
        }
        keys_ = cc_->KeyGen();
        cc_->EvalMultKeyGen(keys_.secretKey);
        if (request.bootstrap_enabled) {
            cc_->EvalBootstrapKeyGen(keys_.secretKey, static_cast<uint32_t>(slots_));
        }

        const std::vector<int32_t> rotations = required_rotation_indices(request_, slots_);
        rotation_key_count_ = static_cast<int>(rotations.size());
        if (!rotations.empty()) {
            cc_->EvalAtIndexKeyGen(keys_.secretKey, rotations);
        }
    }

    Plaintext plain(const std::vector<double>& values) const {
        std::vector<std::complex<double>> packed(slots_, std::complex<double>(0.0, 0.0));
        const std::size_t limit = std::min(values.size(), slots_);
        for (std::size_t i = 0; i < limit; ++i) {
            packed[i] = std::complex<double>(values[i], 0.0);
        }
        return cc_->MakeCKKSPackedPlaintext(packed);
    }

    EncVec encrypt(const std::vector<double>& values, std::size_t logical_size) {
        return {cc_->Encrypt(keys_.publicKey, plain(values)), logical_size};
    }

    EncVec encrypted_zero(std::size_t logical_size) {
        return encrypt(std::vector<double>(slots_, 0.0), logical_size);
    }

    EncVec zero_like(const EncVec& value, std::size_t logical_size) {
        return {cc_->EvalSub(value.ct, value.ct), logical_size};
    }

    std::vector<double> decrypt(const EncVec& value) const {
        Plaintext decrypted;
        cc_->Decrypt(keys_.secretKey, value.ct, &decrypted);
        decrypted->SetLength(static_cast<usint>(slots_));
        const auto packed = decrypted->GetCKKSPackedValue();
        std::vector<double> out(value.logical_size, 0.0);
        for (std::size_t i = 0; i < out.size() && i < packed.size(); ++i) {
            out[i] = packed[i].real();
        }
        return out;
    }

    EncVec add(const EncVec& left, const EncVec& right) {
        return {cc_->EvalAdd(left.ct, right.ct), std::max(left.logical_size, right.logical_size)};
    }

    EncVec sub(const EncVec& left, const EncVec& right) {
        return {cc_->EvalSub(left.ct, right.ct), std::max(left.logical_size, right.logical_size)};
    }

    EncVec add_plain(const EncVec& left, const std::vector<double>& right) {
        Plaintext encoded = plain(right);
        return {cc_->EvalAdd(left.ct, encoded), left.logical_size};
    }

    EncVec mul_plain(const EncVec& left, const std::vector<double>& right) {
        EncVec lhs = maybe_auto_bootstrap(left);
        counts_.ct_pt_mults += 1;
        Plaintext encoded = plain(right);
        return {cc_->EvalMult(lhs.ct, encoded), lhs.logical_size};
    }

    EncVec mul_plain_scalar(const EncVec& left, double scalar) {
        return mul_plain(left, std::vector<double>(slots_, scalar));
    }

    EncVec mul(const EncVec& left, const EncVec& right) {
        EncVec lhs = maybe_auto_bootstrap(left);
        EncVec rhs = maybe_auto_bootstrap(right);
        counts_.ct_ct_mults += 1;
        counts_.relin_count += 1;
        counts_.rescale_count += 1;
        counts_.depth += 1;
        return {cc_->EvalMult(lhs.ct, rhs.ct), std::max(lhs.logical_size, rhs.logical_size)};
    }

    EncVec policy_bootstrap(const EncVec& value, const std::string& label) {
        if (!request_.bootstrap_enabled) {
            throw std::runtime_error("schedule requires bootstrap before " + label + " but bootstrap_enabled is false");
        }
        return bootstrap(value, true);
    }

    EncVec rotate(const EncVec& value, int index) {
        if (index == 0) {
            return value;
        }
        counts_.rotations += 1;
        return {cc_->EvalAtIndex(value.ct, index), value.logical_size};
    }

    std::shared_ptr<std::vector<DCRTPoly>> rotation_digits(const EncVec& value) {
        return cc_->EvalFastRotationPrecompute(value.ct);
    }

    EncVec rotate_fast(
        const EncVec& value,
        int index,
        const std::shared_ptr<std::vector<DCRTPoly>>& digits) {
        if (index == 0) {
            return value;
        }
        counts_.rotations += 1;
        return {cc_->EvalFastRotation(value.ct, static_cast<uint32_t>(index), digits), value.logical_size};
    }

    EncVec sum_slots(const EncVec& value) {
        EncVec out = value;
        for (std::size_t shift = 1; shift < slots_; shift <<= 1U) {
            out = add(out, rotate(out, static_cast<int>(shift)));
        }
        return out;
    }

    std::size_t slots() const {
        return slots_;
    }

    bool use_bsgs() const {
        return request_.linear_kernel != "diagonal";
    }

    bool use_hoisted() const {
        return request_.linear_kernel == "bsgs_hoisted";
    }

    bool fuse_qkv() const {
        return request_.fuse_qkv;
    }

    std::size_t baby_step() const {
        return std::min<std::size_t>(
            std::max<std::size_t>(1, static_cast<std::size_t>(request_.bsgs_baby_step)),
            slots_);
    }

    const std::string& linear_kernel() const {
        return request_.linear_kernel;
    }

    const std::string& packing_strategy() const {
        return request_.packing_strategy;
    }

    const std::string& token_block_size() const {
        return request_.token_block_size;
    }

    int rotation_key_count() const {
        return rotation_key_count_;
    }

    int bootstrap_policy_count() const {
        return bootstrap_policy_count_;
    }

    int bootstrap_auto_count() const {
        return bootstrap_auto_count_;
    }

    double bootstrap_ms() const {
        return bootstrap_ms_;
    }

  private:
    int remaining_levels(const EncVec& value) const {
        return actual_multiplicative_depth_ -
               static_cast<int>(value.ct->GetLevel()) -
               (static_cast<int>(value.ct->GetNoiseScaleDeg()) - 1);
    }

    EncVec maybe_auto_bootstrap(const EncVec& value) {
        if (!request_.bootstrap_enabled || !request_.bootstrap_auto_guard) {
            return value;
        }
        if (remaining_levels(value) > request_.bootstrap_guard_min_levels) {
            return value;
        }
        return bootstrap(value, false);
    }

    EncVec bootstrap(const EncVec& value, bool policy_requested) {
        const auto start = std::chrono::steady_clock::now();
        EncVec refreshed{
            cc_->EvalBootstrap(
                value.ct,
                static_cast<uint32_t>(request_.bootstrap_num_iterations),
                static_cast<uint32_t>(request_.bootstrap_precision)),
            value.logical_size,
        };
        bootstrap_ms_ += elapsed_ms(start);
        counts_.bootstrap_count += 1;
        if (policy_requested) {
            bootstrap_policy_count_ += 1;
        }
        else {
            bootstrap_auto_count_ += 1;
        }
        return refreshed;
    }

    Request request_;
    std::size_t slots_;
    CostCounts& counts_;
    CryptoContext<DCRTPoly> cc_;
    KeyPair<DCRTPoly> keys_;
    int rotation_key_count_ = 0;
    int actual_multiplicative_depth_ = 0;
    int bootstrap_policy_count_ = 0;
    int bootstrap_auto_count_ = 0;
    double bootstrap_ms_ = 0.0;
};

std::vector<double> slice_float_blob(const Blob& source, std::size_t offset, std::size_t count) {
    if (offset + count > source.f64.size()) {
        throw std::runtime_error("slice out of range for blob: " + source.name);
    }
    return std::vector<double>(
        source.f64.begin() + static_cast<std::ptrdiff_t>(offset),
        source.f64.begin() + static_cast<std::ptrdiff_t>(offset + count));
}

std::vector<double> bsgs_diag(
    const Blob& weight,
    std::size_t out_start,
    std::size_t out_logical,
    std::size_t in_start,
    std::size_t out_dim,
    std::size_t in_dim,
    std::size_t slots,
    std::size_t rotation,
    std::size_t giant,
    bool& any) {
    std::vector<double> diag(slots, 0.0);
    any = false;
    for (std::size_t row_slot = 0; row_slot < out_logical; ++row_slot) {
        const std::size_t out_index = out_start + row_slot;
        if (out_index >= out_dim) {
            continue;
        }
        const std::size_t in_slot = (row_slot + rotation) % slots;
        const std::size_t in_index = in_start + in_slot;
        if (in_index >= in_dim) {
            continue;
        }
        const double value = weight.f64[out_index * in_dim + in_index];
        diag[(row_slot + giant) % slots] = value;
        any = any || std::abs(value) > 0.0;
    }
    return diag;
}

std::vector<double> diagonal_diag(
    const Blob& weight,
    std::size_t out_start,
    std::size_t out_logical,
    std::size_t in_start,
    std::size_t in_dim,
    std::size_t slots,
    std::size_t rotation,
    bool& any) {
    std::vector<double> diag(slots, 0.0);
    any = false;
    for (std::size_t row_slot = 0; row_slot < out_logical; ++row_slot) {
        const std::size_t out_index = out_start + row_slot;
        const std::size_t in_slot = (row_slot + rotation) % slots;
        const std::size_t in_index = in_start + in_slot;
        if (in_index >= in_dim) {
            continue;
        }
        const double value = weight.f64[out_index * in_dim + in_index];
        diag[row_slot] = value;
        any = any || std::abs(value) > 0.0;
    }
    return diag;
}

std::vector<double> bias_vector(const Blob& bias, std::size_t out_start, std::size_t out_logical, std::size_t slots) {
    std::vector<double> out(slots, 0.0);
    for (std::size_t i = 0; i < out_logical; ++i) {
        out[i] = bias.f64[out_start + i];
    }
    return out;
}

std::vector<EncVec> linear_diagonal(
    HEEngine& he,
    const std::vector<EncVec>& inputs,
    const Blob& weight,
    const Blob& bias,
    std::size_t out_dim,
    std::size_t in_dim) {
    const std::size_t slots = he.slots();
    const std::size_t out_chunks = (out_dim + slots - 1) / slots;
    const std::size_t in_chunks = (in_dim + slots - 1) / slots;
    if (inputs.size() != in_chunks) {
        throw std::runtime_error("linear input chunk count mismatch for " + weight.name);
    }

    std::vector<EncVec> outputs;
    outputs.reserve(out_chunks);
    for (std::size_t out_chunk = 0; out_chunk < out_chunks; ++out_chunk) {
        const std::size_t out_start = out_chunk * slots;
        const std::size_t out_logical = std::min(slots, out_dim - out_start);
        EncVec acc = he.zero_like(inputs[0], out_logical);
        acc = he.add_plain(acc, bias_vector(bias, out_start, out_logical, slots));

        for (std::size_t in_chunk = 0; in_chunk < in_chunks; ++in_chunk) {
            const std::size_t in_start = in_chunk * slots;
            for (std::size_t rotation = 0; rotation < slots; ++rotation) {
                bool any = false;
                const std::vector<double> diag =
                    diagonal_diag(weight, out_start, out_logical, in_start, in_dim, slots, rotation, any);
                if (!any) {
                    continue;
                }
                const EncVec rotated = rotation == 0 ? inputs[in_chunk] : he.rotate(inputs[in_chunk], static_cast<int>(rotation));
                acc = he.add(acc, he.mul_plain(rotated, diag));
            }
        }
        outputs.push_back(acc);
    }
    return outputs;
}

std::vector<EncVec> linear_bsgs(
    HEEngine& he,
    const std::vector<EncVec>& inputs,
    const Blob& weight,
    const Blob& bias,
    std::size_t out_dim,
    std::size_t in_dim) {
    const std::size_t slots = he.slots();
    const std::size_t out_chunks = (out_dim + slots - 1) / slots;
    const std::size_t in_chunks = (in_dim + slots - 1) / slots;
    const std::size_t baby_step = he.baby_step();
    if (inputs.size() != in_chunks) {
        throw std::runtime_error("linear input chunk count mismatch for " + weight.name);
    }

    std::vector<EncVec> outputs;
    outputs.reserve(out_chunks);
    for (std::size_t out_chunk = 0; out_chunk < out_chunks; ++out_chunk) {
        const std::size_t out_start = out_chunk * slots;
        const std::size_t out_logical = std::min(slots, out_dim - out_start);
        EncVec acc = he.zero_like(inputs[0], out_logical);
        acc = he.add_plain(acc, bias_vector(bias, out_start, out_logical, slots));

        for (std::size_t in_chunk = 0; in_chunk < in_chunks; ++in_chunk) {
            const std::size_t in_start = in_chunk * slots;
            std::vector<EncVec> baby_rotations;
            baby_rotations.reserve(baby_step);
            baby_rotations.push_back(inputs[in_chunk]);
            auto digits = he.use_hoisted() ? he.rotation_digits(inputs[in_chunk]) : nullptr;
            for (std::size_t baby = 1; baby < baby_step; ++baby) {
                baby_rotations.push_back(
                    he.use_hoisted()
                        ? he.rotate_fast(inputs[in_chunk], static_cast<int>(baby), digits)
                        : he.rotate(inputs[in_chunk], static_cast<int>(baby)));
            }

            for (std::size_t giant = 0; giant < slots; giant += baby_step) {
                EncVec partial;
                bool has_partial = false;
                for (std::size_t baby = 0; baby < baby_step && giant + baby < slots; ++baby) {
                    bool any = false;
                    const std::vector<double> diag = bsgs_diag(
                        weight,
                        out_start,
                        out_logical,
                        in_start,
                        out_dim,
                        in_dim,
                        slots,
                        giant + baby,
                        giant,
                        any);
                    if (!any) {
                        continue;
                    }
                    EncVec product = he.mul_plain(baby_rotations[baby], diag);
                    partial = has_partial ? he.add(partial, product) : product;
                    has_partial = true;
                }
                if (!has_partial) {
                    continue;
                }
                acc = he.add(acc, giant == 0 ? partial : he.rotate(partial, static_cast<int>(giant)));
            }
        }
        outputs.push_back(acc);
    }
    return outputs;
}

std::vector<EncVec> linear(
    HEEngine& he,
    const std::vector<EncVec>& inputs,
    const Blob& weight,
    const Blob& bias,
    std::size_t out_dim,
    std::size_t in_dim) {
    return he.use_bsgs()
        ? linear_bsgs(he, inputs, weight, bias, out_dim, in_dim)
        : linear_diagonal(he, inputs, weight, bias, out_dim, in_dim);
}

std::tuple<EncVec, EncVec, EncVec> qkv_linear_fused(
    HEEngine& he,
    const EncVec& input,
    const Blob& q_weight,
    const Blob& q_bias,
    const Blob& k_weight,
    const Blob& k_bias,
    const Blob& v_weight,
    const Blob& v_bias,
    std::size_t hidden) {
    if (!he.use_bsgs()) {
        return {
            linear(he, {input}, q_weight, q_bias, hidden, hidden)[0],
            linear(he, {input}, k_weight, k_bias, hidden, hidden)[0],
            linear(he, {input}, v_weight, v_bias, hidden, hidden)[0],
        };
    }

    const std::size_t slots = he.slots();
    const std::size_t baby_step = he.baby_step();
    EncVec q_acc = he.add_plain(he.zero_like(input, hidden), bias_vector(q_bias, 0, hidden, slots));
    EncVec k_acc = he.add_plain(he.zero_like(input, hidden), bias_vector(k_bias, 0, hidden, slots));
    EncVec v_acc = he.add_plain(he.zero_like(input, hidden), bias_vector(v_bias, 0, hidden, slots));

    std::vector<EncVec> baby_rotations;
    baby_rotations.reserve(baby_step);
    baby_rotations.push_back(input);
    auto digits = he.use_hoisted() ? he.rotation_digits(input) : nullptr;
    for (std::size_t baby = 1; baby < baby_step; ++baby) {
        baby_rotations.push_back(
            he.use_hoisted()
                ? he.rotate_fast(input, static_cast<int>(baby), digits)
                : he.rotate(input, static_cast<int>(baby)));
    }

    for (std::size_t giant = 0; giant < slots; giant += baby_step) {
        EncVec q_partial;
        EncVec k_partial;
        EncVec v_partial;
        bool has_q = false;
        bool has_k = false;
        bool has_v = false;
        for (std::size_t baby = 0; baby < baby_step && giant + baby < slots; ++baby) {
            bool any_q = false;
            bool any_k = false;
            bool any_v = false;
            const std::vector<double> q_diag = bsgs_diag(q_weight, 0, hidden, 0, hidden, hidden, slots, giant + baby, giant, any_q);
            const std::vector<double> k_diag = bsgs_diag(k_weight, 0, hidden, 0, hidden, hidden, slots, giant + baby, giant, any_k);
            const std::vector<double> v_diag = bsgs_diag(v_weight, 0, hidden, 0, hidden, hidden, slots, giant + baby, giant, any_v);
            if (any_q) {
                EncVec product = he.mul_plain(baby_rotations[baby], q_diag);
                q_partial = has_q ? he.add(q_partial, product) : product;
                has_q = true;
            }
            if (any_k) {
                EncVec product = he.mul_plain(baby_rotations[baby], k_diag);
                k_partial = has_k ? he.add(k_partial, product) : product;
                has_k = true;
            }
            if (any_v) {
                EncVec product = he.mul_plain(baby_rotations[baby], v_diag);
                v_partial = has_v ? he.add(v_partial, product) : product;
                has_v = true;
            }
        }
        if (has_q) {
            q_acc = he.add(q_acc, giant == 0 ? q_partial : he.rotate(q_partial, static_cast<int>(giant)));
        }
        if (has_k) {
            k_acc = he.add(k_acc, giant == 0 ? k_partial : he.rotate(k_partial, static_cast<int>(giant)));
        }
        if (has_v) {
            v_acc = he.add(v_acc, giant == 0 ? v_partial : he.rotate(v_partial, static_cast<int>(giant)));
        }
    }
    return {q_acc, k_acc, v_acc};
}

EncVec encrypted_power(
    HEEngine& he,
    const EncVec& x,
    int exponent,
    std::map<int, EncVec>& cache) {
    const auto found = cache.find(exponent);
    if (found != cache.end()) {
        return found->second;
    }
    EncVec out;
    if (exponent % 2 == 0) {
        EncVec half = encrypted_power(he, x, exponent / 2, cache);
        out = he.mul(half, half);
    }
    else {
        EncVec previous = encrypted_power(he, x, exponent - 1, cache);
        out = he.mul(previous, x);
    }
    cache.emplace(exponent, out);
    return out;
}

EncVec gelu_chebyshev(HEEngine& he, const EncVec& x, const GeluApprox& approx) {
    EncVec result = he.encrypted_zero(x.logical_size);
    if (!approx.power_coefficients.empty()) {
        result = he.add_plain(result, std::vector<double>(he.slots(), approx.power_coefficients[0]));
    }
    std::map<int, EncVec> powers;
    powers.emplace(1, x);
    for (std::size_t exponent = 1; exponent < approx.power_coefficients.size(); ++exponent) {
        const double coeff = approx.power_coefficients[exponent];
        if (std::abs(coeff) < 1.0e-18) {
            continue;
        }
        EncVec term = encrypted_power(he, x, static_cast<int>(exponent), powers);
        result = he.add(result, he.mul_plain_scalar(term, coeff));
    }
    return result;
}

EncVec layernorm(
    HEEngine& he,
    const EncVec& x,
    const Blob& gamma_blob,
    const Blob& beta_blob,
    std::size_t hidden,
    const std::string& candidate_id) {
    std::vector<double> gamma(he.slots(), 0.0);
    std::vector<double> beta(he.slots(), 0.0);
    for (std::size_t i = 0; i < hidden; ++i) {
        gamma[i] = gamma_blob.f64[i];
        beta[i] = beta_blob.f64[i];
    }
    if (candidate_id.find("affine.low_cost") != std::string::npos) {
        return he.add_plain(he.mul_plain(x, gamma), beta);
    }

    EncVec mean = he.mul_plain_scalar(he.sum_slots(x), 1.0 / static_cast<double>(hidden));
    EncVec centered = he.sub(x, mean);
    if (candidate_id.find("centered.mid_cost") != std::string::npos) {
        return he.add_plain(he.mul_plain(centered, gamma), beta);
    }

    EncVec centered2 = he.mul(centered, centered);
    EncVec variance = he.mul_plain_scalar(he.sum_slots(centered2), 1.0 / static_cast<double>(hidden));
    EncVec delta = he.add_plain(variance, std::vector<double>(he.slots(), -1.0));
    EncVec delta2 = he.mul(delta, delta);
    EncVec inv_std = he.encrypted_zero(hidden);
    inv_std = he.add_plain(inv_std, std::vector<double>(he.slots(), 1.0));
    inv_std = he.add(inv_std, he.mul_plain_scalar(delta, -0.5));
    inv_std = he.add(inv_std, he.mul_plain_scalar(delta2, 0.375));
    EncVec normalized = he.mul(centered, inv_std);
    return he.add_plain(he.mul_plain(normalized, gamma), beta);
}

EncVec reciprocal_around_public(HEEngine& he, const EncVec& value, double center) {
    const double inv = 1.0 / center;
    EncVec delta = he.add_plain(value, std::vector<double>(he.slots(), -center));
    EncVec delta2 = he.mul(delta, delta);
    EncVec result = he.encrypted_zero(value.logical_size);
    result = he.add_plain(result, std::vector<double>(he.slots(), inv));
    result = he.add(result, he.mul_plain_scalar(delta, -inv * inv));
    result = he.add(result, he.mul_plain_scalar(delta2, inv * inv * inv));
    return result;
}

EncVec exp_approx(HEEngine& he, const EncVec& x, const std::string& candidate_id) {
    EncVec x2 = he.mul(x, x);
    EncVec result = he.encrypted_zero(x.logical_size);
    result = he.add_plain(result, std::vector<double>(he.slots(), 1.0));
    if (candidate_id.find("power.degree2") != std::string::npos) {
        result = he.add(result, he.mul_plain_scalar(x, 2.0));
        result = he.add(result, x2);
        return result;
    }
    result = he.add(result, x);
    result = he.add(result, he.mul_plain_scalar(x2, 0.5));
    if (candidate_id.find("exact.high") != std::string::npos) {
        EncVec x3 = he.mul(x2, x);
        result = he.add(result, he.mul_plain_scalar(x3, 1.0 / 6.0));
    }
    return result;
}

std::vector<double> head_mask(std::size_t slots, std::size_t head_index, std::size_t head_dim) {
    std::vector<double> mask(slots, 0.0);
    const std::size_t start = head_index * head_dim;
    for (std::size_t i = 0; i < head_dim && start + i < slots; ++i) {
        mask[start + i] = 1.0;
    }
    return mask;
}

std::string candidate_for(
    const Manifest& manifest,
    int layer,
    const std::string& op,
    const std::string& name,
    const std::string& fallback);

std::string bootstrap_policy_for(
    const Manifest& manifest,
    int layer,
    const std::string& op,
    const std::string& name);

const GeluApprox& gelu_approx_for(
    const Manifest& manifest,
    int layer,
    const std::string& name);

EncVec maybe_policy_bootstrap(
    HEEngine& he,
    const EncVec& value,
    const Manifest& manifest,
    int layer,
    const std::string& op,
    const std::string& name);

std::vector<EncVec> attention(
    HEEngine& he,
    const std::vector<EncVec>& hidden_states,
    const Blob& attention_mask,
    std::size_t sample_index,
    const Manifest& manifest,
    int layer_index,
    StageTimes& stage_times) {
    const std::size_t seq = static_cast<std::size_t>(manifest.sequence_length);
    const std::size_t hidden = static_cast<std::size_t>(manifest.hidden_size);
    const std::size_t heads = static_cast<std::size_t>(manifest.num_heads);
    if (hidden % heads != 0) {
        throw std::runtime_error("hidden_size must be divisible by num_heads");
    }
    const std::size_t head_dim = hidden / heads;
    const std::string prefix = "layer." + std::to_string(layer_index) + ".attention.";
    const std::string softmax_candidate =
        candidate_for(manifest, layer_index, "softmax", "attention_softmax", "softmax.poly_exp.degree2.v1");

    std::vector<EncVec> q(seq), k(seq), v(seq);
    const auto qkv_start = std::chrono::steady_clock::now();
    for (std::size_t token = 0; token < seq; ++token) {
        if (he.fuse_qkv()) {
            std::tie(q[token], k[token], v[token]) = qkv_linear_fused(
                he,
                hidden_states[token],
                blob(manifest, prefix + "q_lin.weight"),
                blob(manifest, prefix + "q_lin.bias"),
                blob(manifest, prefix + "k_lin.weight"),
                blob(manifest, prefix + "k_lin.bias"),
                blob(manifest, prefix + "v_lin.weight"),
                blob(manifest, prefix + "v_lin.bias"),
                hidden);
        }
        else {
            q[token] = linear(he, {hidden_states[token]}, blob(manifest, prefix + "q_lin.weight"), blob(manifest, prefix + "q_lin.bias"), hidden, hidden)[0];
            k[token] = linear(he, {hidden_states[token]}, blob(manifest, prefix + "k_lin.weight"), blob(manifest, prefix + "k_lin.bias"), hidden, hidden)[0];
            v[token] = linear(he, {hidden_states[token]}, blob(manifest, prefix + "v_lin.weight"), blob(manifest, prefix + "v_lin.bias"), hidden, hidden)[0];
        }
    }
    stage_times.qkv_ms += elapsed_ms(qkv_start);

    const auto attention_start = std::chrono::steady_clock::now();
    std::vector<EncVec> context(seq);
    for (std::size_t query = 0; query < seq; ++query) {
        EncVec token_context = he.zero_like(q[query], hidden);
        for (std::size_t head = 0; head < heads; ++head) {
            const std::vector<double> mask = head_mask(he.slots(), head, head_dim);
            std::vector<EncVec> exp_scores;
            exp_scores.reserve(seq);
            EncVec denom = he.zero_like(q[query], hidden);
            int valid_keys = 0;
            for (std::size_t key = 0; key < seq; ++key) {
                const int public_mask = attention_mask.i32[sample_index * seq + key];
                if (public_mask == 0) {
                    exp_scores.push_back(he.zero_like(q[query], hidden));
                    continue;
                }
                EncVec qh = he.mul_plain(q[query], mask);
                EncVec kh = he.mul_plain(k[key], mask);
                EncVec dot = he.sum_slots(he.mul(qh, kh));
                dot = he.mul_plain_scalar(dot, 1.0 / std::sqrt(static_cast<double>(head_dim)));
                dot = maybe_policy_bootstrap(he, dot, manifest, layer_index, "softmax", "attention_softmax");
                EncVec exp_score = exp_approx(he, dot, softmax_candidate);
                denom = he.add(denom, exp_score);
                exp_scores.push_back(exp_score);
                valid_keys += 1;
            }
            const double center = static_cast<double>(std::max(valid_keys, 1));
            EncVec inv_denom = reciprocal_around_public(he, denom, center);
            EncVec head_context = he.zero_like(q[query], hidden);
            for (std::size_t key = 0; key < seq; ++key) {
                if (attention_mask.i32[sample_index * seq + key] == 0) {
                    continue;
                }
                EncVec weight = he.mul(exp_scores[key], inv_denom);
                EncVec vh = he.mul_plain(v[key], mask);
                head_context = he.add(head_context, he.mul(vh, weight));
            }
            token_context = he.add(token_context, head_context);
        }
        context[query] = linear(he, {token_context}, blob(manifest, prefix + "out_lin.weight"), blob(manifest, prefix + "out_lin.bias"), hidden, hidden)[0];
    }
    stage_times.attention_ms += elapsed_ms(attention_start);
    return context;
}

std::string candidate_for(
    const Manifest& manifest,
    int layer,
    const std::string& op,
    const std::string& name,
    const std::string& fallback) {
    const auto exact = manifest.candidates.find(std::to_string(layer) + "|" + op + "|" + name);
    if (exact != manifest.candidates.end()) {
        return exact->second;
    }
    const auto by_type = manifest.candidates.find(std::to_string(layer) + "|" + op + "|");
    return by_type == manifest.candidates.end() ? fallback : by_type->second;
}

std::string bootstrap_policy_for(
    const Manifest& manifest,
    int layer,
    const std::string& op,
    const std::string& name) {
    const auto exact = manifest.bootstrap_policies.find(std::to_string(layer) + "|" + op + "|" + name);
    return exact == manifest.bootstrap_policies.end() ? "none" : exact->second;
}

const GeluApprox& gelu_approx_for(
    const Manifest& manifest,
    int layer,
    const std::string& name) {
    if (layer < 0) {
        if (!manifest.has_default_gelu) {
            throw std::runtime_error("forward manifest missing default GELU approximation");
        }
        return manifest.default_gelu;
    }
    const auto exact = manifest.gelu_approximations.find(std::to_string(layer) + "|" + name);
    if (exact == manifest.gelu_approximations.end()) {
        throw std::runtime_error("forward manifest missing GELU approximation for layer " +
                                 std::to_string(layer) + " " + name);
    }
    return exact->second;
}

EncVec maybe_policy_bootstrap(
    HEEngine& he,
    const EncVec& value,
    const Manifest& manifest,
    int layer,
    const std::string& op,
    const std::string& name) {
    const std::string policy = bootstrap_policy_for(manifest, layer, op, name);
    if (policy == "none" || policy.empty()) {
        return value;
    }
    if (policy != "bootstrap_before") {
        throw std::runtime_error("unsupported bootstrap_policy: " + policy);
    }
    return he.policy_bootstrap(value, std::to_string(layer) + "|" + op + "|" + name);
}

std::vector<double> run_sample_forward(
    HEEngine& he,
    const Manifest& manifest,
    std::size_t sample_index,
    StageTimes& stage_times) {
    const std::size_t seq = static_cast<std::size_t>(manifest.sequence_length);
    const std::size_t hidden = static_cast<std::size_t>(manifest.hidden_size);
    const std::size_t intermediate = static_cast<std::size_t>(manifest.intermediate_size);
    const Blob& embeddings = blob(manifest, "inputs.embeddings");

    std::vector<EncVec> states(seq);
    const auto encrypt_start = std::chrono::steady_clock::now();
    for (std::size_t token = 0; token < seq; ++token) {
        const std::size_t offset = (sample_index * seq + token) * hidden;
        states[token] = he.encrypt(padded(slice_float_blob(embeddings, offset, hidden), he.slots()), hidden);
    }
    stage_times.encrypt_ms += elapsed_ms(encrypt_start);

    for (int layer = 0; layer < manifest.num_layers; ++layer) {
        const std::string layer_prefix = "layer." + std::to_string(layer) + ".";
        std::vector<EncVec> attn_output =
            attention(he, states, blob(manifest, "inputs.attention_mask"), sample_index, manifest, layer, stage_times);
        for (std::size_t token = 0; token < seq; ++token) {
            EncVec residual = he.add(states[token], attn_output[token]);
            residual = maybe_policy_bootstrap(
                he,
                residual,
                manifest,
                layer,
                "layernorm",
                "attention_layernorm");
            states[token] = layernorm(
                he,
                residual,
                blob(manifest, layer_prefix + "sa_layer_norm.weight"),
                blob(manifest, layer_prefix + "sa_layer_norm.bias"),
                hidden,
                candidate_for(manifest, layer, "layernorm", "attention_layernorm", "layernorm.exact.high.v1"));
        }

        const auto ffn_start = std::chrono::steady_clock::now();
        for (std::size_t token = 0; token < seq; ++token) {
            auto ffn1 = linear(
                he,
                {states[token]},
                blob(manifest, layer_prefix + "ffn.lin1.weight"),
                blob(manifest, layer_prefix + "ffn.lin1.bias"),
                intermediate,
                hidden);
            const GeluApprox& gelu_approx = gelu_approx_for(manifest, layer, "ffn_activation");
            for (auto& chunk : ffn1) {
                chunk = maybe_policy_bootstrap(
                    he,
                    chunk,
                    manifest,
                    layer,
                    "gelu",
                    "ffn_activation");
                chunk = gelu_chebyshev(he, chunk, gelu_approx);
            }
            auto ffn2 = linear(
                he,
                ffn1,
                blob(manifest, layer_prefix + "ffn.lin2.weight"),
                blob(manifest, layer_prefix + "ffn.lin2.bias"),
                hidden,
                intermediate);
            EncVec residual = he.add(states[token], ffn2[0]);
            residual = maybe_policy_bootstrap(
                he,
                residual,
                manifest,
                layer,
                "layernorm",
                "ffn_layernorm");
            states[token] = layernorm(
                he,
                residual,
                blob(manifest, layer_prefix + "output_layer_norm.weight"),
                blob(manifest, layer_prefix + "output_layer_norm.bias"),
                hidden,
                candidate_for(manifest, layer, "layernorm", "ffn_layernorm", "layernorm.exact.high.v1"));
        }
        stage_times.ffn_ms += elapsed_ms(ffn_start);
    }

    const auto classifier_start = std::chrono::steady_clock::now();
    auto pooled = linear(
        he,
        {states[0]},
        blob(manifest, "pre_classifier.weight"),
        blob(manifest, "pre_classifier.bias"),
        hidden,
        hidden);
    pooled[0] = gelu_chebyshev(he, pooled[0], gelu_approx_for(manifest, -1, "classifier_activation"));
    auto logits = linear(
        he,
        pooled,
        blob(manifest, "classifier.weight"),
        blob(manifest, "classifier.bias"),
        static_cast<std::size_t>(manifest.num_labels),
        hidden);
    stage_times.classifier_ms += elapsed_ms(classifier_start);
    const auto decrypt_start = std::chrono::steady_clock::now();
    std::vector<double> decrypted = he.decrypt(logits[0]);
    stage_times.decrypt_ms += elapsed_ms(decrypt_start);
    decrypted.resize(static_cast<std::size_t>(manifest.num_labels));
    return decrypted;
}

std::string forward_result_json(
    const Request& request,
    const Manifest& manifest,
    const CostCounts& counts,
    const std::vector<double>& latencies,
    double accuracy,
    const fs::path& predictions_path,
    const fs::path& logits_path,
    const StageTimes& stage_times) {
    const double mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) /
                        static_cast<double>(std::max<std::size_t>(latencies.size(), 1));
    const double p50 = percentile(latencies, 0.50);
    const double p95 = percentile(latencies, 0.95);
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"case_name\": \"" << json_escape(request.case_name) << "\",\n";
    out << "  \"schedule_name\": \"" << json_escape(request.schedule_name) << "\",\n";
    out << "  \"feasible\": true,\n";
    out << "  \"accuracy\": " << accuracy << ",\n";
    out << "  \"sample_count\": " << manifest.sample_count << ",\n";
    out << "  \"latency_ms\": " << mean << ",\n";
    out << "  \"latency_p50_ms\": " << p50 << ",\n";
    out << "  \"latency_p95_ms\": " << p95 << ",\n";
    out << "  \"predictions_path\": \"" << json_escape(predictions_path.filename().string()) << "\",\n";
    out << "  \"logits_path\": \"" << json_escape(logits_path.filename().string()) << "\",\n";
    out << "  \"error\": \"\",\n";
    out << "  \"cost\": {\n";
    out << "    \"latency_ms\": " << mean << ",\n";
    out << "    \"rotations\": " << counts.rotations << ",\n";
    out << "    \"ct_ct_mults\": " << counts.ct_ct_mults << ",\n";
    out << "    \"ct_pt_mults\": " << counts.ct_pt_mults << ",\n";
    out << "    \"rescale_count\": " << counts.rescale_count << ",\n";
    out << "    \"relin_count\": " << counts.relin_count << ",\n";
    out << "    \"depth\": " << counts.depth << ",\n";
    out << "    \"bootstrap_count\": " << counts.bootstrap_count << ",\n";
    out << "    \"memory_mb\": " << counts.memory_mb << "\n";
    out << "  },\n";
    out << "  \"backend_metadata\": {\n";
    out << "    \"runner\": \"hetune_openfhe_runner\",\n";
    out << "    \"runner_mode\": \"openfhe_distilbert_forward\",\n";
    out << "    \"openfhe_workload\": \"distilbert_encrypted_tensor_forward\",\n";
    out << "    \"linear_kernel\": \"" << json_escape(request.linear_kernel) << "\",\n";
    out << "    \"bsgs_baby_step\": " << request.bsgs_baby_step << ",\n";
    out << "    \"fuse_qkv\": " << (request.fuse_qkv ? "true" : "false") << ",\n";
    out << "    \"packing_strategy\": \"" << json_escape(request.packing_strategy) << "\",\n";
    out << "    \"token_block_size\": \"" << json_escape(request.token_block_size) << "\",\n";
    out << "    \"rotation_key_count\": " << stage_times.rotation_key_count << ",\n";
    out << "    \"keygen_ms\": " << stage_times.keygen_ms << ",\n";
    out << "    \"encrypt_ms\": " << stage_times.encrypt_ms << ",\n";
    out << "    \"qkv_ms\": " << stage_times.qkv_ms << ",\n";
    out << "    \"attention_ms\": " << stage_times.attention_ms << ",\n";
    out << "    \"ffn_ms\": " << stage_times.ffn_ms << ",\n";
    out << "    \"classifier_ms\": " << stage_times.classifier_ms << ",\n";
    out << "    \"decrypt_ms\": " << stage_times.decrypt_ms << ",\n";
    out << "    \"bootstrap_mode\": \"" << (request.bootstrap_enabled ? "executed" : "disabled") << "\",\n";
    out << "    \"bootstrap_ms\": " << stage_times.bootstrap_ms << ",\n";
    out << "    \"bootstrap_policy_count\": " << stage_times.bootstrap_policy_count << ",\n";
    out << "    \"bootstrap_auto_count\": " << stage_times.bootstrap_auto_count << ",\n";
    out << "    \"bootstrap_level_budget\": ["
        << request.bootstrap_level_budget[0] << ", " << request.bootstrap_level_budget[1] << "],\n";
    out << "    \"bootstrap_levels_after\": " << request.bootstrap_levels_after << ",\n";
    out << "    \"privacy_boundary\": \"client_embedding\",\n";
    out << "    \"accuracy_source\": \"native_decrypted_logits\",\n";
    out << "    \"manifest_path\": \"" << json_escape(manifest.path.string()) << "\"\n";
    out << "  }\n";
    out << "}\n";
    return out.str();
}

std::vector<std::vector<double>> run_forward_once(
    const Request& request,
    const Manifest& manifest,
    CostCounts& counts,
    StageTimes& stage_times) {
    const std::size_t slots = next_power_of_two(static_cast<std::size_t>(manifest.hidden_size));
    const auto keygen_start = std::chrono::steady_clock::now();
    HEEngine he(request, slots, counts);
    stage_times.keygen_ms += elapsed_ms(keygen_start);
    stage_times.rotation_key_count = he.rotation_key_count();
    std::vector<std::vector<double>> logits;
    logits.reserve(static_cast<std::size_t>(manifest.sample_count));
    for (std::size_t sample = 0; sample < static_cast<std::size_t>(manifest.sample_count); ++sample) {
        logits.push_back(run_sample_forward(he, manifest, sample, stage_times));
    }
    stage_times.bootstrap_ms = he.bootstrap_ms();
    stage_times.bootstrap_policy_count = he.bootstrap_policy_count();
    stage_times.bootstrap_auto_count = he.bootstrap_auto_count();
    counts.memory_mb = static_cast<double>(slots * manifest.sample_count * manifest.sequence_length * sizeof(double)) /
                       (1024.0 * 1024.0);
    return logits;
}

void write_predictions(
    const Manifest& manifest,
    const std::vector<std::vector<double>>& logits,
    const fs::path& predictions_path,
    const fs::path& logits_path,
    double& accuracy) {
    const Blob& labels = blob(manifest, "inputs.labels");
    int correct = 0;
    std::ofstream pred(predictions_path);
    std::ofstream logit_file(logits_path);
    if (!pred || !logit_file) {
        throw std::runtime_error("failed to open prediction outputs");
    }
    pred << "sample_id,label,prediction";
    logit_file << "sample_id";
    for (int label = 0; label < manifest.num_labels; ++label) {
        pred << ",logit_" << label;
        logit_file << ",logit_" << label;
    }
    pred << "\n";
    logit_file << "\n";

    for (std::size_t sample = 0; sample < logits.size(); ++sample) {
        const auto& row = logits[sample];
        const int prediction = static_cast<int>(std::distance(row.begin(), std::max_element(row.begin(), row.end())));
        const int label = labels.i32[sample];
        correct += prediction == label ? 1 : 0;
        pred << sample << "," << label << "," << prediction;
        logit_file << sample;
        for (double value : row) {
            pred << "," << std::setprecision(10) << value;
            logit_file << "," << std::setprecision(10) << value;
        }
        pred << "\n";
        logit_file << "\n";
    }
    accuracy = logits.empty() ? 0.0 : static_cast<double>(correct) / static_cast<double>(logits.size());
}

void run_schedule_workload(const Request& request, const fs::path& output_path) {
    CostCounts counts;
    if (!load_counts_from_metrics(request.output_dir, request.schedule_name, counts)) {
        counts = load_counts_from_schedule(request.schedule_path);
    }

    std::vector<double> latencies;
    latencies.reserve(static_cast<std::size_t>(request.latency_repetitions));
    for (int rep = 0; rep < request.latency_repetitions; ++rep) {
        latencies.push_back(run_openfhe_workload_once(counts));
    }
    write_file(output_path, workload_result_json(request, counts, latencies));
}

void run_distilbert_forward(const Request& request, const fs::path& output_path) {
    Manifest manifest = load_manifest(request.forward_manifest_path);
    CostCounts counts;
    std::vector<double> latencies;
    std::vector<std::vector<double>> final_logits;
    StageTimes final_stage_times;
    latencies.reserve(static_cast<std::size_t>(request.latency_repetitions));
    for (int rep = 0; rep < request.latency_repetitions; ++rep) {
        CostCounts rep_counts;
        StageTimes rep_stage_times;
        const auto start = std::chrono::steady_clock::now();
        auto logits = run_forward_once(request, manifest, rep_counts, rep_stage_times);
        const auto stop = std::chrono::steady_clock::now();
        if (rep == request.latency_repetitions - 1) {
            final_logits = std::move(logits);
            counts = rep_counts;
            final_stage_times = rep_stage_times;
        }
        latencies.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }

    const fs::path predictions_path = request.output_dir / "predictions.csv";
    const fs::path logits_path = request.output_dir / "logits.csv";
    double accuracy = 0.0;
    write_predictions(manifest, final_logits, predictions_path, logits_path, accuracy);
    write_file(
        output_path,
        forward_result_json(
            request,
            manifest,
            counts,
            latencies,
            accuracy,
            predictions_path,
            logits_path,
            final_stage_times));
}

int main(int argc, char** argv) {
    try {
        fs::path request_path;
        fs::path output_path;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--request" && i + 1 < argc) {
                request_path = argv[++i];
            }
            else if (arg == "--output" && i + 1 < argc) {
                output_path = argv[++i];
            }
        }
        if (request_path.empty() || output_path.empty()) {
            throw std::runtime_error("usage: hetune_openfhe_runner --request request.json --output result.json");
        }

        const Request request = load_request(request_path);
        if (request.runner_mode == "openfhe_distilbert_forward") {
            run_distilbert_forward(request, output_path);
        }
        else if (request.runner_mode == "openfhe_schedule_workload") {
            run_schedule_workload(request, output_path);
        }
        else {
            throw std::runtime_error("unsupported runner_mode: " + request.runner_mode);
        }
        return 0;
    }
    catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }
}
