#ifndef INCLUDE_RL_BUFFERS_TENSOR_AND_POINTER_H_
#define INCLUDE_RL_BUFFERS_TENSOR_AND_POINTER_H_

#include <vector>
#include <memory>

#include "tensor.h"


namespace rl::buffers
{

    template<typename T>
    struct TensorAndObjectBatch
    {
        std::vector<torch::Tensor> tensors;
        std::vector<T> objs;

        int64_t size() { return objs.size(); }
    };

    template<typename T>
    class TensorAndObject
    {
        public:
            TensorAndObject(
                int64_t capacity,
                const std::vector<std::vector<int64_t>> &tensor_shapes,
                const std::vector<torch::TensorOptions> &tensor_options
            ) : tensor{capacity, tensor_shapes, tensor_options}
            {
                obj_data.resize(capacity);
            }

            int64_t size() { return tensor.size(); }

            void clear() { 
                tensor.clear();
            }

            torch::Tensor add(
                const std::vector<torch::Tensor> &tensor_data,
                const std::vector<T> &obj_data
            ) {
                assert (tensor_data.size() > 0);
                auto bs = tensor_data[0].size(0);
                assert(obj_data.size() == bs);

                auto indices = tensor.add(tensor_data);
                assert(indices.size(0) == bs);
                for (int i = 0; i < bs; i++) {
                    this->obj_data[indices.index({i}).item().toLong()] = obj_data[i];
                }

                return indices;
            }

            std::unique_ptr<TensorAndObjectBatch<T>> get(const torch::Tensor &indices) {
                auto n = indices.size(0);
                auto re = std::make_unique<TensorAndObjectBatch<T>>();
                re->tensors = *tensor.get(indices);
                re->objs.reserve(n);

                for (int i = 0; i < n; i++) {
                    re->objs.push_back(obj_data[indices.index({i}).item().toLong()]);
                }

                return re;
            }

            std::unique_ptr<TensorAndObjectBatch<T>> get(const std::vector<int64_t> &indices) {
                return get(torch::tensor(indices, torch::TensorOptions{}.dtype(torch::kLong)));
            }

            std::unique_ptr<TensorAndObjectBatch<T>> get_all() { return get(torch::arange(size())); }

        private:
            Tensor tensor;
            std::vector<T> obj_data;
    };
}

#endif /* INCLUDE_RL_BUFFERS_TENSOR_AND_POINTER_H_ */
