#include <iostream>
#include <cmath>
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"


#include "tensor_utils/q4_npu_eXpress.hpp"

int main(int argc, char* argv[]) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->required(), "Model file");
    arg_utils::parse_options(argc, argv, desc, vm);

    const std::string model_path = vm["model"].as<std::string>();

    std::cout << "Model: " << model_path << std::endl;

    if (!utils::check_file_exists(model_path + "/model.q4nx")) {
        std::cout << "Model file does not exist: " << model_path << std::endl;
        return 1;
    }

    // Load the model
    Q4NX model_weights(model_path);

    // Run the model
    buffer<u8> weight;
    header_print("info", "Loading weights...");
    model_weights.load_weights(weight, "model.layers.0.self_attn.q_proj.weight");

    header_print("info", "Converting weights...");
    buffer<bf16> weight_bf16;
    size_t total_size = weight.size() / model_weights.get_block_size() * model_weights.get_weight_per_chunk();
    int D = sqrt((float)total_size);

    Q4NX::q4nx_dequantize<bf16>(weight_bf16, weight, D);

    utils::print_matrix(weight_bf16, D);

    return 0;
}