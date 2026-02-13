#include <iostream>
#include <cmath>
#define NOMINMAX
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "AutoModel/modeling_lfm2.hpp"
#include "model_list.hpp"
#include "utils/vm_args.hpp"
#include "metrices.hpp"

xrt::device npu_device_global;

// Model-specific factory function for Llama family and DeepSeek_r1_8b
inline std::pair<std::string, std::unique_ptr<AutoModel>> get_lfm2_model(const std::string& model_tag) {
    static std::unordered_set<std::string> lfm2Tags = {
        "lfm2", "lfm2:1.2b", "lfm2:2.6b", "lfm2.5-it:1.2b", "lfm2.5-tk:1.2b", "lfm2-trans:2.6b"
    };

    std::unique_ptr<AutoModel> auto_chat_engine = nullptr;
    std::string new_model_tag = model_tag;

    if (lfm2Tags.count(model_tag))
        auto_chat_engine = std::make_unique<LFM2>(&npu_device_global);
    else {
        new_model_tag = "lfm2:1.2b"; // Default to LFM2 1.2B
        auto_chat_engine = std::make_unique<LFM2>(&npu_device_global);
    }
  
    return std::make_pair(new_model_tag, std::move(auto_chat_engine));
}


// int main(int argc, char* argv[]) {
//     // Set thread priority to low
//     SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
    
//     arg_utils::po::options_description desc("Allowed options");
//     arg_utils::po::variables_map vm;
//     arg_utils::add_default_options(desc);
//     desc.add_options()("model,m", arg_utils::po::value<std::string>()->required(), "Model file");
//     desc.add_options()("Short,s", arg_utils::po::value<bool>()->default_value(true), "Short Prompt");
//     desc.add_options()("Preemption,p", arg_utils::po::value<bool>()->default_value(false), "Preemption");
//     arg_utils::parse_options(argc, argv, desc, vm);

//     std::string tag = vm["model"].as<std::string>();
//     bool short_prompt = vm["Short"].as<bool>();
//     bool preemption = vm["Preemption"].as<bool>();
//     std::cout << "Model: " << tag << std::endl;
//     std::string model_list_path = "model_list.json";
//     std::string exe_dir = ".";
//     model_list model_list(model_list_path, exe_dir);
   
//     header_print("info", "Initializing chat model...");
//     std::string model_path = model_list.get_model_path(tag);
//     nlohmann::json model_info = model_list.get_model_info(tag);

//     LM_Config config;
//     config.from_pretrained(model_path);
//     npu_device_global = xrt::device(0);
    
//     npu_xclbin_manager npu = npu_xclbin_manager(npu_device::device_npu2, &npu_device_global, preemption);
//     // Use model-specific factory
//     auto [actual_tag, chat] = get_lfm2_model(tag);
//     if (actual_tag != tag) {
//         std::cout << "Model tag adjusted to: " << actual_tag << std::endl;
//         model_path = model_list.get_model_path(actual_tag);
//         model_info = model_list.get_model_info(actual_tag);
//     }
   

//     // std::ifstream file(model_path + "/output_sequence.bin", std::ios::binary);
//     std::ifstream file(model_path + "/output.bin", std::ios::binary);
//     if (!file.is_open()) {
//         std::cout << "Failed to open output file" << std::endl;
//         return 1;
//     }
//     file.seekg(0, std::ios::end);
//     size_t size = file.tellg();
//     file.seekg(0, std::ios::beg);
//     buffer<bf16> reference(size / sizeof(bf16));
//     file.read(reinterpret_cast<char*>(reference.data()), size);
//     tensor_2d<bf16> reference_tensor(reference, 2048);
//     lfm2_npu lfm2_npu_engine(config, &npu, 8192);
//     std::unique_ptr<Q4NX> q4nx = std::make_unique<Q4NX>(model_path);
//     lfm2_npu_engine.load_weights(*q4nx);
//     std::vector<int> tokens = {1,     6,  6423,   708,  7347,   975,  2793,   803,  6586,   988,
//         779,  4632,  7349,  1090, 15195,   523,  4112,  1014,  1112,   936,
//         797,  8682,  9629,   523,  5947,  1517, 57015,   523,  1517,   997,
//       20954,   511,   875,  5019,   597,   523,     7,   708,     6, 64015,
//         708};
    
//     int L_got = size / sizeof(bf16) / 2048;
//     buffer<bf16> y = lfm2_npu_engine.prefill(tokens, nullptr);
//     tensor_2d<bf16> y_tensor(y, 2048);
//     buffer<float> errors(L_got);
//     for (int i = 0; i < L_got; i++){
//         auto error_metrics = get_error_metrics(y_tensor[i], reference_tensor[i]);
//         errors[i] = error_metrics.CosineSimilarity;
//     }
//     std::cout << "Cosine Similarity: " << std::endl;
//     utils::print_matrix(errors, 1, 64);

//     return 0;
// }

// Wait for final test

int main(int argc, char* argv[]) {
    // Set thread priority to low
    // SetConsoleOutputCP(CP_UTF8);
    // SetConsoleCP(CP_UTF8);
    // SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
    
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->required(), "Model file");
    desc.add_options()("Short,s", arg_utils::po::value<bool>()->default_value(true), "Short Prompt");
    desc.add_options()("Preemption,p", arg_utils::po::value<bool>()->default_value(false), "Preemption");
    arg_utils::parse_options(argc, argv, desc, vm);

    std::string tag = vm["model"].as<std::string>();
    bool short_prompt = vm["Short"].as<bool>();
    bool preemption = vm["Preemption"].as<bool>();
    std::cout << "Model: " << tag << std::endl;
    std::string model_list_path = "model_list.json";
    std::string exe_dir = ".";
    model_list model_list(model_list_path, exe_dir);


   
    header_print("info", "Initializing chat model...");
    std::string model_path = model_list.get_model_path(tag);
    nlohmann::json model_info = model_list.get_model_info(tag);

    npu_device_global = xrt::device(0); 
    // Use model-specific factory
    auto [actual_tag, chat] = get_lfm2_model(tag);
    if (actual_tag != tag) {
        std::cout << "Model tag adjusted to: " << actual_tag << std::endl;
        model_path = model_list.get_model_path(actual_tag);
        model_info = model_list.get_model_info(actual_tag);
    }
   
    chat->load_model(model_path, model_info, 131072, preemption);
    chat->set_topk(1);
    chat_meta_info_t meta_info;
    lm_uniform_input_t uniformed_input;


    if (short_prompt) {
        uniformed_input.prompt = "Who are you? ";
        std::cout << "Prompt: " << uniformed_input.prompt << std::endl;
        std::cout << "Response: ";
        chat->start_total_timer();
        std::string response = chat->generate_with_prompt(meta_info, uniformed_input, 1024, std::cout);
        chat->stop_total_timer();
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << chat->show_profile() << std::endl;
        uniformed_input.prompt = "What is C. elegans?";
        std::cout << "Prompt: " << uniformed_input.prompt << std::endl;
        std::cout << "Response: ";
        chat->start_total_timer();
        response = chat->generate_with_prompt(meta_info, uniformed_input, 1024, std::cout);
        chat->stop_total_timer();
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << chat->show_profile() << std::endl;

        uniformed_input.prompt = "How many cells does it have?";
        std::cout << "Prompt: " << uniformed_input.prompt << std::endl;
        std::cout << "Response: ";
        chat->start_total_timer();
        response = chat->generate_with_prompt(meta_info, uniformed_input, 1024, std::cout);
        chat->stop_total_timer();
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << chat->show_profile() << std::endl;
    }
    else{
        std::ifstream file("../../../../test.txt", std::ios::binary);
        if (!file.is_open()) {
            std::cout << "Failed to open prompt file" << std::endl;
            return 1;
        }
        uniformed_input.prompt = "";
        file.seekg(0, std::ios::end);
        uniformed_input.prompt.resize(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(uniformed_input.prompt.data(), uniformed_input.prompt.size());
        file.close();
        std::cout << "Prompt: " << uniformed_input.prompt << std::endl;
        std::cout << "Response: ";
        chat->start_total_timer();
        std::string response = chat->generate_with_prompt(meta_info, uniformed_input, 128, std::cout);
        chat->stop_total_timer();
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << chat->show_profile() << std::endl;
    }
   
    // std::pair<std::string, std::vector<int>> history = chat->get_history();
    // std::cout << "History length: " << history.second.size() << std::endl;

    return 0;
}
