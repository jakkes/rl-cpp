#include "rl/agents/ppo/trainers/basic.h"

#include <thread_pool.hpp>

#include "rl/buffers/tensor_and_pointer.h"
#include "rl/policies/constraints/base.h"

#include "rl/cpputils.h"


using namespace torch::indexing;
using namespace rl;

namespace rl::agents::ppo::trainers
{
    static
    torch::Tensor policy_loss(torch::Tensor A, torch::Tensor old_probs, torch::Tensor new_probs, float eps)
    {
        auto pr = new_probs / old_probs;
        auto clipped = pr.clamp(1-eps, 1+eps);
        pr = pr * A;
        clipped = clipped * A;
        return - torch::min(clipped, pr).mean();
    }

    static
    torch::Tensor compute_deltas(torch::Tensor rewards, torch::Tensor V, torch::Tensor not_terminals, float discount)
    {
        return rewards + discount * not_terminals * V.index({"...", Slice(1, None)}).detach() - V.index({"...", Slice(None, -1)});
    }

    static
    torch::Tensor compute_advantages(torch::Tensor deltas, torch::Tensor not_terminals, float discount, float gae_discount)
    {
        float d = discount * gae_discount;
        auto A = torch::empty_like(deltas);

        A.index_put_({"...", -1}, deltas.index({"...", -1}));
        for (int k = A.size(1) - 2; k > -1; k--) {
            A.index_put_({"...", k}, deltas.index({"...", k}) + d * not_terminals.index({"...", k}) * A.index({"...", k + 1}));
        }

        return A;
    }

    struct Sequences {
        std::vector<torch::Tensor> states{};
        std::vector<torch::Tensor> actions{};
        std::vector<torch::Tensor> rewards{};
        std::vector<torch::Tensor> not_terminals{};
        std::vector<torch::Tensor> action_probabilities{};
        std::vector<torch::Tensor> state_values{};
        std::vector<std::vector<std::shared_ptr<policies::constraints::Base>>> constraints{};

        Sequences(int batchsize)
        {
            states.reserve(batchsize);
            actions.reserve(batchsize);
            rewards.reserve(batchsize);
            not_terminals.reserve(batchsize);
            action_probabilities.reserve(batchsize);
            state_values.reserve(batchsize);
            constraints.reserve(batchsize);
        }
    };

    struct Sequence {
        std::vector<torch::Tensor> states{};
        std::vector<torch::Tensor> actions{};
        std::vector<torch::Tensor> rewards{};
        std::vector<torch::Tensor> not_terminals{};
        std::vector<torch::Tensor> action_probabilities{};
        std::vector<torch::Tensor> state_values{};
        std::vector<std::shared_ptr<policies::constraints::Base>> constraints{};

        Sequence(int length)
        {
            states.reserve(length + 1);
            actions.reserve(length);
            rewards.reserve(length);
            not_terminals.reserve(length);
            action_probabilities.reserve(length);
            state_values.reserve(length);
            constraints.reserve(length+1);
        }
    };

    static
    std::unique_ptr<Sequence> run_sequence(
        std::shared_ptr<env::Base> env,
        agents::ppo::Module *model,
        int length
    )
    {
        auto re = std::make_unique<Sequence>(length);

        auto state = env->state();
        if (env->is_terminal()) state = env->reset();

        re->states.push_back(state->state);
        re->constraints.push_back(state->action_constraint);

        return re;
    }

    static
    std::unique_ptr<Sequences> run_sequences(
        const std::vector<std::shared_ptr<env::Base>> &envs,
        agents::ppo::Module *model,
        int length,
        thread_pool *pool
    )
    {
        auto re = std::make_unique<Sequences>(envs.size());
        return re;
    }

    static
    void run_sequence(env::Base *env, agents::ppo::Module *model, int K, Sequences *out, int i)
    {
        std::vector<torch::Tensor> states;
        states.reserve(K+1);
    }

    Basic::Basic(
        std::shared_ptr<agents::ppo::Module> model,
        std::unique_ptr<torch::optim::Optimizer> optimizer,
        std::shared_ptr<env::Factory> env_factory,
        const BasicOptions &options
    ) : model{model}, optimizer{std::move(optimizer)},
        env_factory{env_factory}, options{options}
    {}

    void Basic::_run()
    {
        thread_pool pool(options.env_workers);

        while (true)
        {
            
        }

        std::vector<env::Base> envs{};
        envs.reserve(options.envs);
        for (int i = 0; i < options.envs; i++) {
            envs.push_back(*env_factory->get());
        }
    }

    template<class Rep, class Period>
    void Basic::run(std::chrono::duration<Rep, Period> duration)
    {
        auto start = std::chrono::steady_clock::now();
        auto end = start + duration;

        _run();

        while (std::chrono::steady_clock::now() < end)
        {
            return;
        }
    }
}
