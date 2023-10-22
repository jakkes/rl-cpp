#ifndef RL_AGENTS_DQN_MODULE_H_
#define RL_AGENTS_DQN_MODULE_H_


#include <torch/torch.h>

namespace rl::agents::dqn {

    
    /**
     * @brief Base class for DQN modules.
     * 
     * When creating a network, one must inherit from this class and
     * torch::nn::Clonebale.
     * 
     * Example:
     * class Network : public rl::agents::dqn::Module, public torch::nn::Cloneable<Network>
     * {
     *    public:
     *       Network() = default;
     *       ~Network() = default;
     * 
     *       void reset() override {
     *           linear1 = register_module("linear1", torch::nn::Linear{10, 10})
     *       }
     * 
     *       torch::Tensor forward(const torch::Tensor &x) const override
     *       {
     *          return torch::relu(linear1->forward(x));
     *       }
     *
     *    private:
     *       torch::nn::Linear linear1;
     * };
    */
    class Module : public virtual torch::nn::Module
    {
        public:
            virtual ~Module() = default;

            /**
             * @brief Forward pass of the network.
             * 
             * @param x Input tensor
             * @return torch::Tensor Output tensor
             */
            virtual torch::Tensor forward(const torch::Tensor &x) = 0;
    };
}

#endif /* RL_AGENTS_DQN_MODULE_H_ */
