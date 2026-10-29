// mnemosyne_inference.cpp
// C++ inference for MNEMOSYNE-MNS using ONNX Runtime
// Build: g++ -std=c++17 mnemosyne_inference.cpp -lonnxruntime -o mnemosyne_inference

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include <cmath>

using json = nlohmann::json;

class MnemosyneRanker {
public:
    MnemosyneRanker(const std::string& model_dir) {
        // Load ONNX Runtime
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Mnemosyne");
        
        // Load E5 encoder
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        std::string encoder_path = model_dir + "/e5_encoder.onnx";
        e5_session_ = std::make_unique<Ort::Session>(*env_, encoder_path.c_str(), session_options);
        
        // Load scaler parameters
        std::ifstream scaler_file(model_dir + "/scaler.json");
        json scaler_json;
        scaler_file >> scaler_json;
        scaler_mean_ = scaler_json["mean"].get<std::vector<float>>();
        scaler_scale_ = scaler_json["scale"].get<std::vector<float>>();
        
        // Load ranker parameters
        std::ifstream ranker_file(model_dir + "/ranker.json");
        json ranker_json;
        ranker_file >> ranker_json;
        ranker_coef_ = ranker_json["coef"].get<std::vector<float>>();
        ranker_intercept_ = ranker_json["intercept"].get<float>();
    }
    
    std::vector<float> encode(const std::string& text) {
        // Tokenize (simplified - use proper tokenizer in production)
        std::vector<int64_t> input_ids = tokenize(text);
        std::vector<int64_t> attention_mask(input_ids.size(), 1);
        
        // Prepare input tensors
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};
        
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, input_ids.data(), input_ids.size(), 
            input_shape.data(), input_shape.size()
        );
        Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, attention_mask.data(), attention_mask.size(),
            input_shape.data(), input_shape.size()
        );
        
        // Run inference
        const char* input_names[] = {"input_ids", "attention_mask"};
        const char* output_names[] = {"last_hidden_state"};
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_ids_tensor));
        input_tensors.push_back(std::move(attention_mask_tensor));
        
        auto output_tensors = e5_session_->Run(
            Ort::RunOptions{nullptr},
            input_names, input_tensors.data(), 2,
            output_names, 1
        );
        
        // Mean pooling
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int seq_len = output_shape[1];
        int hidden_dim = output_shape[2];
        
        std::vector<float> pooled(hidden_dim, 0.0f);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < hidden_dim; ++j) {
                pooled[j] += output_data[i * hidden_dim + j];
            }
        }
        for (auto& v : pooled) v /= seq_len;
        
        // Normalize
        float norm = 0.0f;
        for (auto v : pooled) norm += v * v;
        norm = std::sqrt(norm);
        for (auto& v : pooled) v /= norm;
        
        return pooled;
    }
    
    float rank(const std::vector<float>& features) {
        // Standardize features
        std::vector<float> scaled_features(features.size());
        for (size_t i = 0; i < features.size(); ++i) {
            scaled_features[i] = (features[i] - scaler_mean_[i]) / scaler_scale_[i];
        }
        
        // Logistic regression: score = dot(coef, features) + intercept
        float score = ranker_intercept_;
        for (size_t i = 0; i < ranker_coef_.size(); ++i) {
            score += ranker_coef_[i] * scaled_features[i];
        }
        
        return score;
    }
    
    std::vector<int64_t> tokenize(const std::string& text) {
        // Simplified tokenization - use sentencepiece or HuggingFace tokenizer in production
        // This is a placeholder
        std::vector<int64_t> tokens = {101}; // [CLS]
        for (char c : text) {
            tokens.push_back(static_cast<int64_t>(c));
        }
        tokens.push_back(102); // [SEP]
        return tokens;
    }
    
private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> e5_session_;
    std::vector<float> scaler_mean_;
    std::vector<float> scaler_scale_;
    std::vector<float> ranker_coef_;
    float ranker_intercept_;
};

int main() {
    try {
        MnemosyneRanker ranker("./runs/onnx_export");
        
        // Example usage
        std::string query = "如何配置代理？";
        std::string memory = "在 git config 中设置 http.proxy";
        
        auto query_emb = ranker.encode("query: " + query);
        auto memory_emb = ranker.encode("passage: " + memory);
        
        // Compute cosine similarity
        float e5_cosine = 0.0f;
        for (size_t i = 0; i < query_emb.size(); ++i) {
            e5_cosine += query_emb[i] * memory_emb[i];
        }
        
        // Construct feature vector (simplified - add other 12 features)
        std::vector<float> features = {e5_cosine, 0.8f, 0.5f, 0.3f, 2.1f, 3.4f, 
                                       0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f};
        
        float score = ranker.rank(features);
        std::cout << "Ranking score: " << score << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
