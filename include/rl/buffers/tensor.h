#ifndef INCLUDE_RL_BUFFERS_TENSOR_H_
#define INCLUDE_RL_BUFFERS_TENSOR_H_


#include <vector>
#include <mutex>
#include <atomic>

#include <torch/torch.h>


namespace rl::buffers
{
    class Tensor
    {
        public:
            Tensor(
                int64_t capacity,
                const std::vector<std::vector<int64_t>> &tensor_shapes,
                const std::vector<torch::TensorOptions> &tensor_options
            );

            torch::Tensor add(const std::vector<torch::Tensor> &data);

            int64_t size() const;
            std::unique_ptr<std::vector<torch::Tensor>> get(torch::Tensor indices);
            std::unique_ptr<std::vector<torch::Tensor>> get(const std::vector<int64_t> &indices);

        private:
            const int64_t capacity;
            std::mutex lock{};
            
            std::vector<torch::Tensor> data;
            std::atomic<int64_t> memory_index{0};
            bool looped{false};
    };
}

#endif /* INCLUDE_RL_BUFFERS_TENSOR_H_ */
