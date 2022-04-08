#include "inference_batch.h"


using namespace rl;

namespace rl::agents::ppo::trainers::seed_impl
{
    InferenceBatch::InferenceBatch(
        std::shared_ptr<rl::agents::ppo::Module> model,
        InferenceOptions *options
    ) :
    model{model}, options{options}
    {
        states.reserve(options->batchsize);
        constraints.reserve(options->batchsize);
    }

    int64_t InferenceBatch::add_inference_request(const env::State &state)
    {
        std::lock_guard lock{add_mtx};

        if (has_executed()) throw InferenceBatchObsolete{};
        assert(!is_full());

        states.push_back(state.state);
        constraints.push_back(state.action_constraint);

        if (is_full()) {
            execute();
        } else if (is_empty()) {
            start_timer();
        }

        size++;
        return size - 1;
    }

    torch::Tensor InferenceBatch::get_inference_result(int64_t label)
    {
        
    }
}