#ifndef INCLUDE_RL_BUFFERS_TENSOR_AND_POINTER_H_
#define INCLUDE_RL_BUFFERS_TENSOR_AND_POINTER_H_

#include <vector>
#include <memory>

#include "tensor.h"


namespace rl::buffers
{

    template<typename T>
    struct TensorAndPointerBatch
    {
        std::unique_ptr<std::vector<torch::Tensor>> tensors;
        std::vector<std::shared_ptr<T>> ptrs;

        int64_t size() { return ptrs.size(); }
    };

    template<typename T>
    class TensorAndPointer
    {
        public:
            TensorAndPointer(
                int64_t capacity,
                const std::vector<std::vector<int64_t>> &tensor_shapes,
                const std::vector<torch::TensorOptions> &tensor_options
            ) : tensor{capacity, tensor_shapes, tensor_options}
            {
                ptr_data.resize(capacity);
            }

            int64_t size() { return tensor.size(); }

            void clear() { 
                std::lock_guard<std::mutex> guard{lock};
                tensor.clear();
            }

            torch::Tensor add(
                const std::vector<torch::Tensor> &tensor_data,
                const std::vector<std::shared_ptr<T>> &ptr_data
            ) {
                assert (tensor_data.size() > 0);
                auto bs = tensor_data[0].size(0);
                assert(ptr_data.size() == bs);

                std::lock_guard<std::mutex> lock_guard{lock};
                auto indices = tensor.add(tensor_data);
                assert(indices.size(0) == bs);
                for (int i = 0; i < bs; i++) {
                    this->ptr_data[indices.index({i}).item().toLong()] = ptr_data[i];
                }

                return indices;
            }

            std::unique_ptr<TensorAndPointerBatch<T>> get(const torch::Tensor &indices) {
                std::lock_guard<std::mutex> lock_guard{lock};
                
                auto n = indices.size(0);
                auto re = std::make_unique<TensorAndPointerBatch<T>>();
                re->tensors = tensor.get(indices);
                re->ptrs.reserve(n);

                for (int i = 0; i < n; i++) {
                    re->ptrs.push_back(ptr_data[indices.index({i}).item().toLong()]);
                }

                return re;
            }

            std::unique_ptr<TensorAndPointerBatch<T>> get(const std::vector<int64_t> &indices) {
                return get(torch::tensor(indices, torch::TensorOptions{}.dtype(torch::kLong)));
            }

        private:
            std::mutex lock{};
            Tensor tensor;
            std::vector<std::shared_ptr<T>> ptr_data;
    };
}

#endif /* INCLUDE_RL_BUFFERS_TENSOR_AND_POINTER_H_ */
