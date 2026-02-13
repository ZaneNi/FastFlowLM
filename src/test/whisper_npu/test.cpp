#include <iostream>
#include <cmath>
#define NOMINMAX
#include <windows.h>
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "model_list.hpp"
#include "utils/vm_args.hpp"
#include "metrices.hpp"
#include "models/whisper/modeling_whisper.hpp"


int main(int argc, char* argv[]) {
    // // Set thread priority to low
    // SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
    
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->required(), "Model file");
    desc.add_options()("Preemption,p", arg_utils::po::value<bool>()->default_value(false), "Preemption");
    desc.add_options()("Audio,a", arg_utils::po::value<std::string>()->required(), "Audio file");
    desc.add_options()("Timestamp,t", arg_utils::po::value<bool>()->default_value(false), "Timestamp");
    desc.add_options()("Return_time_stamp,r", arg_utils::po::value<bool>()->default_value(false), "Return time stamp");
    arg_utils::parse_options(argc, argv, desc, vm);

    std::string tag = vm["model"].as<std::string>();
    bool preemption = vm["Preemption"].as<bool>();
    std::string audio_filename = vm["Audio"].as<std::string>();
    bool timestamp = vm["Timestamp"].as<bool>();
    bool return_time_stamp = vm["Return_time_stamp"].as<bool>();
    std::cout << "Model: " << tag << std::endl;
    std::string model_list_path = "model_list.json";
    std::string exe_dir = ".";
    model_list model_list(model_list_path, exe_dir);
   
    header_print("info", "Initializing chat model...");
    std::string model_path = model_list.get_model_path(tag);
    nlohmann::json model_info = model_list.get_model_info(tag);

    xrt::device npu_device_global = xrt::device(0);

    Whisper whisper(&npu_device_global);

    std::string audio_path = "C:\\Users\\alfred\\Downloads\\nvidia.mp3";

    whisper.load_model(model_path, model_info, preemption);
    header_print("info", "Loading audio...");
    bool ret = whisper.load_audio(audio_path);
    std::pair<std::string, std::string> result = whisper.generate(Whisper::whisper_task_type_t::e_transcribe, timestamp, return_time_stamp, std::cout);
    std::cout << std::endl;
    std::cout << "Language detected: " << result.second << std::endl;
    std::cout << "Result: " << result.first << std::endl;

    // ret = whisper.load_audio(audio_path);
    // result = whisper.generate(Whisper::whisper_task_type_t::e_translate, timestamp, return_time_stamp, std::cout);
    // std::cout << std::endl;
    // std::cout << "Language detected: " << result.second << std::endl;
    // std::cout << "Result: " << result.first << std::endl;

    // std::vector<uint8_t> audio_data_raw;
    // std::ifstream file(audio_path, std::ios::binary);
    // if (!file.is_open()) {
    //     std::cout << "Failed to open file: " << audio_path << std::endl;
    //     return 1;
    // }
    // file.seekg(0, std::ios::end);
    // size_t size = file.tellg();
    // file.seekg(0, std::ios::beg);
    // audio_data_raw.resize(size);
    // file.read(reinterpret_cast<char*>(audio_data_raw.data()), size);
    // file.close();
    // whisper.load_audio(audio_data_raw);

    // std::cout << "Audio Loaded!" << std::endl;
    // result = whisper.generate(Whisper::whisper_task_type_t::e_translate, timestamp, return_time_stamp, std::cout);
    // std::cout << std::endl;
    // std::cout << "Language detected: " << result.second << std::endl;
    // std::cout << "Result: " << result.first << std::endl;


    // // Use model-specific factory
    // // auto [actual_tag, chat] = get_gpt_oss_model(tag);
    // auto actual_tag = tag; // for now, ignore checking
    // if (actual_tag != tag) {
    //     std::cout << "Model tag adjusted to: " << actual_tag << std::endl;
    //     model_path = model_list.get_model_path(actual_tag);
    //     model_info = model_list.get_model_info(actual_tag);
    // }
   
    // std::cout << "Model path: " << model_path << std::endl;
    // std::cout << "Model info: " << model_info.dump() << std::endl;

    // std::string audio_filename = model_path + "/nvidia.mp3";
    // std::vector<float> audio_data = read_audio(audio_filename);
    // std::cout << "Audio data size: " << audio_data.size() << std::endl;
    // buffer<float> audio_data_buffer(audio_data);
    // buffer<float> mel_fb_buffer(128 * 201);
    // mel_fb_buffer.from_file(model_path + "/mel_filter.bin");

    // utils::print_matrix(audio_data_buffer, audio_data_buffer.size());

    // buffer<float> audio_reference(audio_data_buffer.size());
    // audio_reference.from_file(model_path + "/audio_array.bin");

    // print_error_metrics(get_error_metrics(audio_data_buffer, audio_reference));
    // std::vector<float> audio_reference_copy(audio_reference.size());
    // for (int i = 0; i < audio_reference.size(); i++) {
    //     audio_reference_copy[i] = audio_reference[i];
    // }


    // time_utils::time_point start = time_utils::now();
    // std::vector<float> log_mel = preprocess_audio(audio_data);
    // time_utils::time_point stop = time_utils::now();
    // std::cout << "Time: " << time_utils::duration_ms(start, stop).first << " ms" << std::endl;
    // std::cout << "Log mel size: " << log_mel.size() << std::endl;
    
    // buffer<bf16> reference_mel(384000);

    // std::string reference_mel_filename = model_path + "/input_features.bin";
    // reference_mel.from_file(reference_mel_filename);
    // buffer<bf16> mel_bf16(reference_mel.size());
    // for (int i = 0; i < reference_mel.size(); i++) {
    //     mel_bf16[i] = bf16_t(log_mel[i]);
    // }
    // print_error_metrics(get_error_metrics(mel_bf16, reference_mel));
    // utils::print_matrix(mel_bf16, 3000);

    return 0;
}
