#ifndef INCLUDE_RL_BUFFERS_TENSOR_AND_POINTER_H_
#define INCLUDE_RL_BUFFERS_TENSOR_AND_POINTER_H_

#include <vector>
#include <memory>

#include "tensor.h"


namespace rl::buffers
{

    /**
     * @brief A batch of samples collected from a `TensorAndObject` buffer.
     * 
     * One batch may hold multiple batched tensors and one batch of objects. The first
     * dimension of each tensor is the batch dimension. Objects are batched using 
     * `std::vector<T>`.
     * 
     * @tparam T Object type.
     */
    template<typename T>
    struct TensorAndObjectBatch
    {
        // Tensors, where the first dimension in each tensor is the batch dimension.
        std::vector<torch::Tensor> tensors;
        // Objects.
        std::vector<T> objs;

        // Size of batch.
        inline int64_t size() { return objs.size(); }
    };

    /**
     * @brief FIFO buffer of tensors and objects.
     * 
     * This class wraps the `rl::buffers::Tensor` class and extends it, allowing for
     * combining tensors and objects in one sample. For details on how tensors are
     * handled, see the documentation on `rl::buffers::Tensor`.
     * 
     * @tparam T Type of object.
     */
    template<typename T>
    class TensorAndObject
    {
        public:
            /**
             * @brief Construct a new TensorAndObject buffer.
             * 
             * @param capacity Buffer capacity
             * @param tensor_shapes Shapes of tensors
             * @param tensor_options Tensor options
             */
            TensorAndObject(
                int64_t capacity,
                const std::vector<std::vector<int64_t>> &tensor_shapes,
                const std::vector<torch::TensorOptions> &tensor_options
            ) : tensor{capacity, tensor_shapes, tensor_options}
            {
                obj_data.resize(capacity);
            }

            /**
             * @return int64_t Number of samples contained in the buffer.
             */
            inline int64_t size() { return tensor.size(); }

            /**
             * @brief Clears the contents of the buffer.
             */
            inline void clear() { 
                tensor.clear();
            }

            /**
             * @brief Adds a batch of samples to the buffer.
             * 
             * @param tensor_data Tensors to be added, with their first dimension being
             * across samples.
             * @param obj_data Objects.
             * @return torch::Tensor Buffer locations of added samples.
             */
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

            /**
             * @brief Returns a batch of samples from the buffer.
             * 
             * @param indices Data locations to be returned.
             * @return std::unique_ptr<TensorAndObjectBatch<T>> Batch of samples.
             */
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

            /**
             * @brief Returns a batch of samples from the buffer.
             * 
             * @param indices Data locations to be returned.
             * @return std::unique_ptr<TensorAndObjectBatch<T>> Batch of samples.
             */
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
