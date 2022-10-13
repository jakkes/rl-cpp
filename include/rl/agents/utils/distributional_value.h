#ifndef RL_AGENTS_UTILS_DISTRIBUTIONAL_LOSS_H_
#define RL_AGENTS_UTILS_DISTRIBUTIONAL_LOSS_H_


#include <torch/torch.h>

namespace rl::agents::utils
{
    class DistributionalValue
    {
        public:
            DistributionalValue() = default;

            DistributionalValue(
                const torch::Tensor &atoms,
                const torch::Tensor &logits
            ) : atoms_{atoms}, logits_{logits}
            {
                v_min_ = atoms.index({0}).item().toFloat();
                v_max_ = atoms.index({-1}).item().toFloat();
            }

            inline
            torch::Tensor mean() const {
                return (logits_.softmax(-1) * atoms_).sum(-1);
            }

            inline
            torch::Tensor atoms() const { return atoms_; }

            inline
            torch::Tensor logits() const { return logits_; }

            inline
            float v_max() const { return v_max_; }

            inline
            float v_min() const { return v_min_; }
        
        private:
            torch::Tensor atoms_, logits_;
            float v_max_, v_min_;
    };

    torch::Tensor distributional_value_loss(
        const torch::Tensor &current_logits,
        const torch::Tensor &rewards,
        const torch::Tensor &not_terminals,
        const torch::Tensor &next_logits,
        const torch::Tensor &atoms,
        float discount
    );
}

#endif /* RL_AGENTS_UTILS_DISTRIBUTIONAL_LOSS_H_ */
