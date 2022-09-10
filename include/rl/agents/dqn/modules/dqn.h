#ifndef RL_AGENTS_DQN_MODULES_DQN_H_
#define RL_AGENTS_DQN_MODULES_DQN_H_


#include <rl/option.h>

#include "base.h"


namespace rl::agents::dqn::modules
{
    class DQNOutput : public BaseOutput
    {
        public:
            DQNOutput(const torch::Tensor &values) : values{values} {}

            inline
            const torch::Tensor value() const override {
                return values;
            }

            inline
            void apply_mask(const rl::policies::constraints::CategoricalMask &masks) {
                auto mask_tensor = ~masks.mask();
                values = values.index_put({mask_tensor}, torch::zeros({mask_tensor.sum().item().toLong()}, values.options()) - INFINITY);
            }

            torch::Tensor loss(
                const torch::Tensor &actions,
                const torch::Tensor &rewards,
                const torch::Tensor &not_terminals,
                const BaseOutput &next_output,
                float discount
            ) override;
        
        private:
            torch::Tensor values;
    };

    /**
     * @brief Basic DQN module.
     * 
     */
    class DQN : public Base
    {
        public:
            /**
             * @brief Implementation of the forward pass.
             * 
             * @param states states
             * @return std::unique_ptr<DQNOutput> output
             */
            virtual 
            std::unique_ptr<DQNOutput> forward_impl(const torch::Tensor &states) = 0;

            inline
            std::unique_ptr<BaseOutput> forward(const torch::Tensor &states) override {
                return forward_impl(states);
            }
    };
}

#endif /* RL_AGENTS_DQN_MODULES_DQN_H_ */
