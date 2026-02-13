#include <iostream>
#include <cmath>
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "AutoModel/modeling_gemma3.hpp"
#include "model_list.hpp"

xrt::device npu_device_global;

// Model-specific factory function for Gemma family only
inline std::pair<std::string, std::unique_ptr<AutoModel>> get_gemma_model(const std::string& model_tag) {
    static std::unordered_set<std::string> gemma3_vlm_tags = {
        "gemma3:4b", 
        "medgemma", "medgemma:4b"
    };

    std::unique_ptr<AutoModel> auto_chat_engine = nullptr;
    std::string new_model_tag = model_tag;
    
  
    if (gemma3_vlm_tags.count(model_tag))
        auto_chat_engine = std::make_unique<Gemma3>(&npu_device_global);
    else {
        new_model_tag = "gemma3:4b"; // Default to text-only model
        auto_chat_engine = std::make_unique<Gemma3>(&npu_device_global);
    }
  
    return std::make_pair(new_model_tag, std::move(auto_chat_engine));
}

int main(int argc, char* argv[]) {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
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
    auto [actual_tag, chat] = get_gemma_model(tag);
    if (actual_tag != tag) {
        std::cout << "Model tag adjusted to: " << actual_tag << std::endl;
        model_path = model_list.get_model_path(actual_tag);
        model_info = model_list.get_model_info(actual_tag);
    }
   
    chat->load_model(model_path, model_info, -1, preemption);
    chat_meta_info_t meta_info;
    lm_uniform_input_t uniformed_input;
    std::string response;

    if (short_prompt) {
        uniformed_input.prompt = "Describe this?";
        uniformed_input.images.push_back("../../../../tb_files/panda.png");
        uniformed_input.image_payload_types.push_back(input_payload_type_t::FILE_NAME);
        std::cout << "Prompt: " << uniformed_input.prompt << std::endl;
        std::cout << "Response: ";
        chat->start_total_timer();
        // std::string response = chat->generate_with_prompt(meta_info, uniformed_input, 1024, std::cout);
        if (chat->insert(meta_info, uniformed_input))
        {
            response = chat->generate(meta_info, 2048, std::cout);
        }
        chat->stop_total_timer();
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << chat->show_profile() << std::endl;
       
        uniformed_input.prompt = "Can I have one as a pet?";
        uniformed_input.images.clear();
        uniformed_input.image_payload_types.clear();
        std::cout << "Prompt: " << uniformed_input.prompt << std::endl;
        std::cout << "Response: ";
        chat->start_total_timer();
        // std::string response = chat->generate_with_prompt(meta_info, uniformed_input, 1024, std::cout);
        if (chat->insert(meta_info, uniformed_input))
        {
            response = chat->generate(meta_info, 2048, std::cout);
        }
        chat->stop_total_timer();
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << chat->show_profile() << std::endl;

        
        uniformed_input.prompt = "Can I catch one from the wild?";
        uniformed_input.images.clear();
        uniformed_input.image_payload_types.clear();
        std::cout << "Prompt: " << uniformed_input.prompt << std::endl;
        std::cout << "Response: ";
        chat->start_total_timer();
        // std::string response = chat->generate_with_prompt(meta_info, uniformed_input, 1024, std::cout);
        if (chat->insert(meta_info, uniformed_input))
        {
            response = chat->generate(meta_info, 2048, std::cout);
        }
        chat->stop_total_timer();
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << chat->show_profile() << std::endl;
    }
    else{
        std::ifstream file("../../../../prompt.txt", std::ios::binary);
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
        std::string response = chat->generate_with_prompt(meta_info, uniformed_input, 1024, std::cout);
        chat->stop_total_timer();
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << chat->show_profile() << std::endl;
    }
   
    std::pair<std::string, std::vector<int>> history = chat->get_history();
    std::cout << "History length: " << history.second.size() << std::endl;

    for (int i = 0; i < history.second.size(); i++){
        std::cout << history.second[i] << " " << std::flush;
    }

    std::cout << std::endl;

    return 0;
}