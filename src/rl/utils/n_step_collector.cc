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

        if (looped)
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

        states[i] = state;
        actions[i] = action;
        for (int j = 0; j < n; j++) {
            int k = (i - j + n) % n;
            rewards[j][k] = reward * std::pow(discount, k);
        }
        i++;
        if (i >= n) {
            looped = true;
            i = 0;
        }

        if (terminal)
        {
            int N = looped ? n : i;
            int offset = out.size();
            out.resize(offset + N);
            for (int j = 0; j < N; j++)
            {
                out[offset + j].action = actions[j];
                out[offset + j].reward = 0.0f;
                for (int k = 0; k < n; k++) out[offset + j].reward += rewards[j][k];
                out[offset + j].next_state = state;
                out[offset + j].terminal = true;
                out[offset + j].state = states[j];

                for (int k = 0; k < n; k++) rewards[j][k] = 0.0f;
            }

            i = 0;
            looped = false;
        }

        return out;
    }


}
