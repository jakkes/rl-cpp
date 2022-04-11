#include "inference_batch.h"


using namespace rl;

namespace rl::agents::ppo::trainers::seed_impl
{
    InferenceBatch::InferenceBatch(
        std::shared_ptr<rl::agents::ppo::Module> model,
        const InferenceOptions *options
    ) :
    model{model}, options{options}
    {
        states.reserve(options->batchsize);
        constraints.reserve(options->batchsize);
    }

    InferenceBatch::~InferenceBatch()
    {
        if (timer_thread.joinable()) timer_thread.join();
    }

    int64_t InferenceBatch::add_inference_request(const env::State &state)
    {
        std::lock_guard lock{add_mtx};

        if (has_executed()) throw InferenceBatchObsolete{};
        assert(!is_full());

        states.push_back(state.state);
        constraints.push_back(state.action_constraint);

        size++;

        if (is_full()) {
            execute();
        } else if (is_empty()) {
            start_timer();
        }

        return size - 1;
    }

    std::unique_ptr<InferenceResult> InferenceBatch::get_inference_result(int64_t label)
    {
        std::unique_lock lock{execute_mtx};
        execute_cv.wait(lock, [this] () { return has_executed(); });

        auto out = std::make_unique<InferenceResult>();
        out->action = result_actions.index({label});
        out->value = result_values.index({label});
        out->action_probability = result_probabilities.index({label});
        return out;
    }

    void InferenceBatch::execute()
    {
        std::lock_guard lock{execute_mtx};
        if (has_executed()) return;

        torch::NoGradGuard no_grad{};
        auto states = torch::stack(this->states);

        auto model_output = model->forward(states);
        model_output->policy->include(
            policies::constraints::stack(this->constraints)
        );

        result_actions = model_output->policy->sample();
        result_values = model_output->value;
        result_probabilities = model_output->policy->prob(result_actions);

        executed = true;
        execute_cv.notify_all();
    }

    void InferenceBatch::start_timer()
    {
        timer_thread = std::thread(&InferenceBatch::timer, this);
    }

    void InferenceBatch::timer()
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(options->max_delay_ms));
        execute();
    }
}