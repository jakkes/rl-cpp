#ifndef INCLUDE_RL_BUFFERS_TENSOR_H_
#define INCLUDE_RL_BUFFERS_TENSOR_H_


#include <vector>
#include <mutex>
#include <atomic>

#include <torch/torch.h>


namespace rl::buffers
{
    /**
     * @brief FIFO buffer holding data in tensors.
     * 
     * This buffer accepts data samples that consist of multiple tensors of different
     * sizes and types, so long all samples share these.
     */
    class Tensor
    {
        public:
            /**
             * @brief Construct a new Tensor buffer.
             * 
             * @param capacity Buffer capacity.
             * @param tensor_shapes List of shapes that each tensor in one sample take.
             * @param tensor_options List of options describing the type of each tensor
             * in one sample.
             */
            Tensor(
                int64_t capacity,
                const std::vector<std::vector<int64_t>> &tensor_shapes,
                const std::vector<torch::TensorOptions> &tensor_options
            );

            /**
             * @brief Adds a __batch__ of samples to the buffer.
             * 
             * @param data List of batched tensors, where the first dimension denotes
             * sample (batch dimension). All tensors must be of shape (N, *), where *
             * denote the shape given in the constructor.
             * @return torch::Tensor Storage locations of added samples, shape (N).
             */
            torch::Tensor add(const std::vector<torch::Tensor> &data);

            /**
             * @brief Clears the buffer of all its content.
             */
            void clear();

            /**
             * @return int64_t Number of elements currently stored in the buffer.
             */
            int64_t size() const;

            /**
             * @brief Collects and returns a batch of samples from the buffer.
             * 
             * @param indices Location of samples to collect, shape (N)
             * @return std::unique_ptr<std::vector<torch::Tensor>> List of tensors,
             * where each tensor is of shape (N, *) with * denoting the shape given
             * in the constructor.
             */
            std::unique_ptr<std::vector<torch::Tensor>> get(torch::Tensor indices);

            /**
             * @brief Collects and returns a batch of samples from the buffer.
             * 
             * @param indices Location of samples to collect.
             * @return std::unique_ptr<std::vector<torch::Tensor>> List of tensors,
             * where each tensor is of shape (N, *) with * denoting the shape given
             * in the constructor, and N denoting the length of `indices`.
             */
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
