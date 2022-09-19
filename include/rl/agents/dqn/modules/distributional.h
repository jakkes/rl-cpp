#ifndef RL_AGENTS_DQN_MODULES_DISTRIBUTIONAL_H_
#define RL_AGENTS_DQN_MODULES_DISTRIBUTIONAL_H_


#include "base.h"

namespace rl::agents::dqn::modules
{
    class DistributionalOutput : public BaseOutput
    {
        public:
            DistributionalOutput(
                const torch::Tensor &distributions,
                const torch::Tensor &atoms,
                float v_min,
                float v_max
            );

            const torch::Tensor value() const override;

            void apply_mask(const torch::Tensor &mask) override;

            torch::Tensor loss(
                const torch::Tensor &actions,
                const torch::Tensor &rewards,
                const torch::Tensor &not_terminals,
                const BaseOutput &next_output,
                const torch::Tensor &next_actions,
                float discount
            ) override;
        
        private:
            const torch::Tensor distributions, atoms;
            const float v_min, v_max;
            torch::Tensor inverted_mask;
            int64_t n_atoms;
            float dz;
            bool mask_set{false};
    };

    /**
     * @brief Distributional DQN module
     * 
     */
    class Distributional : public Base
    {
        public:
            /**
             * @brief Implementation of the forward pass.
             * 
             * @param states states
             * @return std::unique_ptr<DistributionalOutput> output
             */
            virtual 
            std::unique_ptr<DistributionalOutput> forward_impl(const torch::Tensor &states) = 0;

            inline
            std::unique_ptr<BaseOutput> forward(const torch::Tensor &states) override {
                return forward_impl(states);
            }
    };
}

#endif /* RL_AGENTS_DQN_MODULES_DISTRIBUTIONAL_H_ */
