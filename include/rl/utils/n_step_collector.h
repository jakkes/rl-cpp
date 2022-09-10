#ifndef RL_UTILS_N_STEP_H_
#define RL_UTILS_N_STEP_H_


#include <vector>

#include <torch/torch.h>

#include <rl/env/observation.h>


namespace rl::utils
{
    struct NStepCollectorTransition
    {
        std::shared_ptr<rl::env::State> state;
        torch::Tensor action;
        float reward;
        bool terminal;
        std::shared_ptr<rl::env::State> next_state;
    };

    /**
     * @brief Utility class for building n step rewards.
     * 
     */
    class NStepCollector
    {
        public:
            NStepCollector(int n, float discount);

            std::vector<NStepCollectorTransition> step(
                std::shared_ptr<rl::env::State> state,
                torch::Tensor action,
                float reward,
                bool terminal
            );

        private:
            const int n;
            const float discount;
            int i{0};
            bool looped{false};

            std::vector<std::vector<float>> rewards;
            std::vector<std::shared_ptr<rl::env::State>> states;
            std::vector<torch::Tensor> actions;
    };
}

#endif /* RL_UTILS_N_STEP_H_ */
