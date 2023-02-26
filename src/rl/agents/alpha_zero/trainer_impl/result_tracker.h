#ifndef RL_CPP_SRC_RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_RESULT_TRACKER_H_
#define RL_CPP_SRC_RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_RESULT_TRACKER_H_

#include <mutex>

#include <torch/torch.h>
#include <rl/logging/client/base.h>

namespace trainer_impl
{
    class ResultTracker
    {
        public:
            ResultTracker(std::shared_ptr<rl::logging::client::Base> logger);
        
            void report(
                const torch::Tensor &episode_lengths,
                const torch::Tensor &actions,
                const torch::Tensor &rewards
            );

        private:
            std::shared_ptr<rl::logging::client::Base> logger;
            float best_reward{-INFINITY};
            std::mutex mtx{};
    };
}

#endif /* RL_CPP_SRC_RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_RESULT_TRACKER_H_ */
