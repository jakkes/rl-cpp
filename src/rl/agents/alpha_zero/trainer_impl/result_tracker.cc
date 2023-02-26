#include "result_tracker.h"

#include <sstream>


using namespace torch::indexing;

namespace trainer_impl
{
    ResultTracker::ResultTracker(std::shared_ptr<rl::logging::client::Base> logger)
    : logger{logger} {}

    void ResultTracker::report(
        const torch::Tensor &episode_lengths,
        const torch::Tensor &actions,
        const torch::Tensor &rewards
    )
    {
        if (!logger) {
            return;
        }

        auto total_rewards = rewards.sum(1);
        auto max_reward = total_rewards.max().item().toFloat();

        std::lock_guard lock{mtx};
        if (max_reward <= best_reward) {
            return;
        }

        best_reward = max_reward;

        auto i = rewards.argmax().item().toLong();
        auto best_actions = actions.index({
            i,
            Slice(None, episode_lengths.index({i}).item().toLong())
        });

        std::stringstream sstream{};

        sstream << "New best observed reward: " << max_reward << ", through action sequence: ";
        for (int64_t i = 0; i < best_actions.size(0); i++) {
            sstream << best_actions.index({i}).item().toLong() << " ";
        }
        sstream << "\n";
        logger->log_text("AlphaZero/Best episode", sstream.str());
    }
}
