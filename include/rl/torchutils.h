#ifndef UTILS_TORCH_H_
#define UTILS_TORCH_H_


#include <unordered_map>
#include <string>

#include <torch/torch.h>

namespace rl::torchutils
{
    bool is_int_dtype(const torch::Tensor &data);

    bool is_bool_dtype(const torch::Tensor &data);


    /**
     * @brief Base functionality for building classes holding tensors.
     * 
     * Tensors should be registered on creation using the `register_tensor` method.
     * 
     */
    class TensorHolder
    {
        private:
            std::unordered_map<std::string, torch::Tensor*> tensors{};
            std::unordered_map<std::string, torch::Tensor> tensor_storage{};
        
        protected:

            /**
             * @brief Registers a tensor to the module.
             * 
             * @param name Name of tensor
             * @param tensor Tensor instance
             * @return torch::Tensor** 
             */
            torch::Tensor **register_tensor(const std::string &name, torch::Tensor tensor);

        public:

            /**
             * @brief Moves all registered tensors to the given device.
             * 
             * @param device New device.
             */
            void to(torch::Device device);
    };
}

#endif /* UTILS_TORCH_H_ */
