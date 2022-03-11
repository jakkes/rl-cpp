#ifndef RL_BUFFERS_STATE_TRANSITION_H_
#define RL_BUFFERS_STATE_TRANSITION_H_

#include <vector>
#include <memory>
#include <mutex>
#include <atomic>

#include <torch/torch.h>

#include "rl/env/state.h"


namespace rl::buffers
{

    struct StateTransitionOptions
    {
        torch::Device device = torch::kCPU;

        auto &device_(torch::Device device) { this->device = device; return *this; }
    };

    struct StateTransitionBatch
    {
        std::shared_ptr<rl::env::State> states;
        torch::Tensor rewards;
        torch::Tensor terminals;
        std::shared_ptr<rl::env::State> next_states;

        int64_t size() { return rewards.size(0); }
    };

    class StateTransition{
        public:
            StateTransition(int64_t capacity, const StateTransitionOptions &options = {});

            void add(
                std::shared_ptr<rl::env::State> state,
                float reward,
                bool terminal,
                std::shared_ptr<rl::env::State> next_state
            );
            void add(
                const std::vector<std::shared_ptr<rl::env::State>> &states,
                const std::vector<float> &rewards,
                const std::vector<bool> &terminals,
                const std::vector<std::shared_ptr<rl::env::State>> &next_states
            );

            int64_t size() const;
            std::unique_ptr<StateTransitionBatch> get(torch::Tensor indices);
            std::unique_ptr<StateTransitionBatch> get(const std::vector<int64_t> &indices);

        private:
            const StateTransitionOptions options;
            const int64_t capacity;
            std::mutex lock{};

            std::vector<std::shared_ptr<rl::env::State>> states;
            torch::Tensor rewards;
            torch::Tensor terminals;
            std::vector<std::shared_ptr<rl::env::State>> next_states;

            std::atomic<int64_t> memory_index{0};
            bool looped{false};
    };
}

#endif /* RL_BUFFERS_STATE_TRANSITION_H_ */