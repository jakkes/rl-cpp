#ifndef RL_ENV_CONSTRAINTS_BASE_H_
#define RL_ENV_CONSTRAINTS_BASE_H_

#include <memory>
#include <vector>
#include <functional>

#include <torch/torch.h>



namespace rl::policies::constraints
{
    class Base : public torch::nn::Module
    {
        public:
            virtual ~Base() = default;
            virtual torch::Tensor contains(const torch::Tensor &x) const = 0;
            virtual std::unique_ptr<Base> index(const std::vector<torch::indexing::TensorIndex> &indexing) const = 0;
            virtual std::unique_ptr<Base> stack(const std::vector<std::shared_ptr<Base>> &constraints) const = 0;

            template<typename T>
            T &as_type() {
                return dynamic_cast<T&>(*this);
            }

            template<typename T>
            bool is_type() {
                return dynamic_cast<T*>(this);
            }
    };
}

#endif /* RL_ENV_CONSTRAINTS_BASE_H_ */
