#include "rl/utils/n_step_collector.h"


namespace rl::utils
{
    NStepCollector::NStepCollector(int n, float discount)
    : n{n}, discount{discount}
    {
        rewards.resize(n);
        for (auto &x : rewards) x.resize(n);
        states.resize(n);
        actions.resize(n);
    }

    std::vector<NStepCollectorTransition> NStepCollector::step(
        std::shared_ptr<rl::env::State> state,
        torch::Tensor action,
        float reward,
        bool terminal
    )
    {
        std::vector<NStepCollectorTransition> out{};

        if (looped && !terminal)
        {
            out.resize(1);
            out[0].action = actions[i];
            out[0].reward = 0.0f;
            for (int j = 0; j < n; j++) out[0].reward += rewards[i][j];
            out[0].next_state = state;
            out[0].terminal = false;
            out[0].state = states[i];

            // Prepare for next rewards
            for (int j = 0; j < n; j++) rewards[i][j] = 0.0f;
        }
        else if (terminal)
        {
            int N = looped ? n : i;
            out.resize(N);
            for (int j = 0; j < N; j++)
            {
                out[j].action = actions[j];
                out[j].reward = 0.0f;
                for (int k = 0; k < n; k++) out[j].reward += rewards[j][k];
                out[j].next_state = state;
                out[j].terminal = true;
                out[j].state = states[j];

                for (int k = 0; k < n; k++) rewards[j][k] = 0.0f;
            }

            i = 0;
            looped = false;
        }

        states[i] = state;
        actions[i] = action;
        for (int j = 0; j < n; j++) {
            rewards[j][(i - j) % n] = reward * std::pow(discount, (i - j) % n);
        }
        i++;
        if (i >= n) {
            looped = true;
            i = 0;
        }
        return out;
    }


}
