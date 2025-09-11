/// \file deepseek.hpp
/// \brief deepseek class
/// \author FastFlowLM Team
/// \date 2025-09-01
/// \version 0.9.9
/// \note This is a source file for the deepseek class

#pragma once
#include "AutoModel/automodel.hpp"

/************              DeepSeek_r1_8b            **************/
class DeepSeek_r1_8b : public AutoModel {
private:
    std::string current_model = "DeepSeek_r1_8b";

    int think_marker_id;

    void setup_tokenizer(std::string model_path);

public:
    DeepSeek_r1_8b(unsigned int device_id);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1) override;
    //void toggle_enable_think() override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages) override;
};

/************              DeepSeek_r1_0528_8b            **************/
class DeepSeek_r1_0528_8b : public AutoModel {
private:
    std::string current_model = "DeepSeek_r1_0528_8b";

    int think_marker_id;

    void setup_tokenizer(std::string model_path);

public:
    DeepSeek_r1_0528_8b(unsigned int device_id);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1) override;
    //void toggle_enable_think() override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages) override;
};
