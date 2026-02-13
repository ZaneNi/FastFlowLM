#include "image/image_reader.hpp"
#include "utils/utils.hpp"
#include "metrices.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
        std::cout << "Example: " << argv[0] << " C:\\path\\to\\image.jpg" << std::endl;
        return 1;
    }

    std::string imagePath = argv[1];
    
    // Test 1: load jpg image
    { 
        std::cout << "Test 1: load jpg image" << std::endl;
        bytes image = load_image("../../../iceCream.jpg");
        if (image.size() == 0) {
            std::cerr << "Failed to load image!" << std::endl;
            return 1;
        }
        
        std::cout << "Image loaded successfully!" << std::endl;
        std::cout << "Size: " << image.size() << " bytes (896x896x3 RGB24)" << std::endl;

        if (save_image("../../../output_image_test1.ppm", image)) {
            std::cout << "Saved PPM image successfully!" << std::endl;
        } else {
            std::cerr << "Failed to save image!" << std::endl;
            return 1;
        }
        std::cout << "Test 1 passed!" << std::endl;
        std::cout << "--------------------------------" << std::endl;
    }
        
    // Test 2: load png image
    {
        std::cout << "Test 2: load png image" << std::endl;
        bytes image = load_image("../../../monke-ai.jpg");
        if (image.size() == 0) {
            std::cerr << "Failed to load image!" << std::endl;
            return 1;
        }


        if (save_image("../../../output_image_test2.ppm", image)) {
            std::cout << "Saved PPM image successfully!" << std::endl;
        } else {
            std::cerr << "Failed to save image!" << std::endl;
            return 1;
        }
        std::cout << "Image loaded successfully!" << std::endl;
        std::cout << "Size: " << image.size() << " bytes (896x896x3 RGB24)" << std::endl;
        buffer<bf16> pixel_values = preprocess_image(image);
        std::cout << "pixel_values.size(): " << pixel_values.size() << std::endl;
        buffer<float> pixel_values_ref(pixel_values.size());

        pixel_values_ref.from_file("../../../pixel_values.bin");
        buffer<bf16> pixel_values_ref_bf16(pixel_values_ref.size());
        for (int i = 0; i < pixel_values_ref.size(); i++) {
            pixel_values_ref_bf16[i] = bf16_t(pixel_values_ref[i]);
        }
        print_error_metrics(get_error_metrics(pixel_values, pixel_values_ref_bf16));
        std::cout << "ref" << std::endl;
        utils::print_matrix(pixel_values_ref_bf16, 256);
        std::cout << "pixel_values" << std::endl;
        utils::print_matrix(pixel_values, 256);
        float mean_ref = 0;
        float max_ref = 0;
        float min_ref = 0;
        for (int i = 0; i < pixel_values_ref_bf16.size(); i++) {
            mean_ref += pixel_values_ref_bf16[i].as_float();
            max_ref = std::max(max_ref, pixel_values_ref_bf16[i].as_float());
            min_ref = std::min(min_ref, pixel_values_ref_bf16[i].as_float());
        }
        mean_ref /= pixel_values_ref_bf16.size();
        float mean_pixel_values = 0;
        for (int i = 0; i < pixel_values.size(); i++) {
            mean_pixel_values += pixel_values[i].as_float();
        }
        mean_pixel_values /= pixel_values.size();
        float max_pixel_values = 0;
        float min_pixel_values = 0;
        for (int i = 0; i < pixel_values.size(); i++) {
            max_pixel_values = std::max(max_pixel_values, pixel_values[i].as_float());
            min_pixel_values = std::min(min_pixel_values, pixel_values[i].as_float());
        }
        std::cout << "mean_ref: " << mean_ref << std::endl;
        std::cout << "max_ref: " << max_ref << std::endl;
        std::cout << "min_ref: " << min_ref << std::endl;
        std::cout << "mean_pixel_values: " << mean_pixel_values << std::endl;
        std::cout << "max_pixel_values: " << max_pixel_values << std::endl;
        std::cout << "min_pixel_values: " << min_pixel_values << std::endl;
        std::cout << "Test 2 passed!" << std::endl;
        std::cout << "--------------------------------" << std::endl;
    }

    bytes raw_image;
    std::ifstream file("../../../resized_image.bin", std::ios::binary);
    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(raw_image.data()), raw_image.size());
    }
    else {
        std::cerr << "Failed to open file!" << std::endl;
        return 1;
    }
    // get the size of the raw_image
    file.seekg(0, std::ios::end);
    int size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::cout << "size: " << size / 3 / 896 << std::endl;
    raw_image.resize(size);
    file.read(reinterpret_cast<char*>(raw_image.data()), size);
    file.close();

    save_image("../../../resized_image.ppm", raw_image);
    buffer<bf16> pixel_values_resized = preprocess_image(raw_image);
    std::cout << "pixel_values_resized.size(): " << pixel_values_resized.size() << std::endl;

    return 0;
    // Test 3: load jpg from base64 encoded string
    {
        std::cout << "Test 3: load jpg from base64 encoded string" << std::endl;

        std::ifstream file("../../../jpg_base64.txt");
        std::stringstream buffer;
        bytes image;
        if (file.is_open()) {
            buffer << file.rdbuf();
            std::string base64_string = buffer.str();
            image = load_image_base64(base64_string);
        }
        else {
            std::cerr << "Failed to open file!" << std::endl;
            return 1;
        }
        std::cout << "Image size: " << image.size() << std::endl;
        // save image as ppm
        if (save_image("../../../output_image_test3.ppm", image)) {
            std::cout << "Saved PPM image successfully!" << std::endl;
        } else {
            std::cerr << "Failed to save image!" << std::endl;
            return 1;
        }
        std::cout << "Test 3 passed!" << std::endl;
        std::cout << "--------------------------------" << std::endl;
    }
   
    // Test 4: load png from base64 encoded string
    {
        std::cout << "Test 4: load png from base64 encoded string" << std::endl;

        std::ifstream file("../../../png_base64.txt");
        std::stringstream buffer;
        bytes image;
        if (file.is_open()) {
            buffer << file.rdbuf();
            std::string base64_string = buffer.str();
            image = load_image_base64(base64_string);
        }
        else {
            std::cerr << "Failed to open file!" << std::endl;
            return 1;
        }
        std::cout << "Image size: " << image.size() << std::endl;
        // save image as ppm
        if (save_image("../../../output_image_test4.ppm", image)) {
            std::cout << "Saved PPM image successfully!" << std::endl;
        } else {
            std::cerr << "Failed to save image!" << std::endl;
            return 1;
        }
        std::cout << "Test 4 passed!" << std::endl;
        std::cout << "--------------------------------" << std::endl;
    }
    
    return 0;
} 