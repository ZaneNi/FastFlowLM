/// \file Qwen2VL.hpp
/// \brief Qwen2VL class
/// \author FastFlowLM Team
/// \date 2025-09-03
/// \version 0.9.24
/// \note This is a source file for the Qwen2VL class

#pragma once
#include "AutoModel/automodel.hpp"
#include "metrices.hpp"


#include "typedef.hpp"
#include "image_process_utils/imageproc.hpp"
#include "image_process_utils/imageprocAVX512.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "base64.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
// FFmpeg includes for image processing only
extern "C" {
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
}


//Parameter for QWEN2IMAGE 
constexpr unsigned int QWEN2_PATCH_SIZE = 14;
constexpr unsigned int QWEN2_IMAGE_MERGE_SIZE=2;
constexpr unsigned int QWEN2_SPATIAL_MERGE_SIZE=2;
constexpr unsigned int QWEN2_SHORTEST_EDGE = 3136;
constexpr unsigned int QWEN2_LONGEST_EDGE = 12845056;
constexpr float QWEN2_VISION_RESCALE_FACTOR = 0.00392156862745098;
constexpr float QWEN2_VISION_RESCALE_IMAGE_MEAN_R = 0.48145466f;
constexpr float QWEN2_VISION_RESCALE_IMAGE_MEAN_G =  0.4578275f;
constexpr float QWEN2_VISION_RESCALE_IMAGE_MEAN_B = 0.40821073f;
constexpr float QWEN2_VISION_RESCALE_IMAGE_STD_R = 0.26862954f;
constexpr float QWEN2_VISION_RESCALE_IMAGE_STD_G = 0.26130258f;
constexpr float QWEN2_VISION_RESCALE_IMAGE_STD_B = 0.27577711f;
constexpr unsigned int QWEN2_WINDOW_ATTENTION_PIXEL_SIZE = 122;
constexpr unsigned int QWEN2_TEMPORAL_PATCH_SIZE = 2;
constexpr unsigned int QWEN2_MERGE_SIZE = 2;


typedef struct {
    int height;
    int width;
    int height_resized;  // assigned by image preprocessing
    int width_resized;
    int grid_h;
    int grid_w;

    std::vector<uint8_t> _data;

} qwen2vl_image_t;



typedef struct {
    std::vector<qwen2vl_image_t> images;
    std::vector<bf16> _data__processed;    
    unsigned int num_images;
}qwen2vl_image_payload_t;




/************              Qwen2VL_4b            **************/
class Qwen2VL : public AutoModel {
private:

    void setup_tokenizer(std::string model_path);
    
    // Image processing functionality
    static bool ffmpeg_initialized;
    void initialize_ffmpeg();
    void resolve_source_format_and_range(AVPixelFormat input_format,
                                        AVPixelFormat &resolved_format,
                                        int &src_full_range,
                                        AVColorRange frame_color_range,
                                        AVCodecID codec_id);
    qwen2vl_image_t load_image(const std::string& filename);
    qwen2vl_image_t load_image_base64(const std::string& base64_string);
    

    int debug_count= 0;
    void smart_resize(
    int height, int width,
    int& h_bar,int& w_bar,
    int factor,
    int min_pixels,
    int max_pixels);
    
    void preprocess_image(qwen2vl_image_t& image,  std::vector<bf16> &pixel_values);

public:
    Qwen2VL(xrt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false) override;
    //void toggle_enable_think() override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object()) override;
};
