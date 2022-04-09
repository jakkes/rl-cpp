#include "inference.h"


namespace rl::agents::ppo::trainers::seed_impl
{



    Inference::Inference(
        std::shared_ptr<rl::agents::ppo::Module> model,
        const InferenceOptions &options
    ) :
    model{model}, options{options}
    {}

    std::shared_ptr<InferenceBatch> Inference::get_batch()
    {
        std::lock_guard lock{current_batch_mtx};
        if (!current_batch || current_batch->has_executed()) {
            current_batch = std::make_shared<InferenceBatch>(model, &options);
        }
        return current_batch;
    }

    std::unique_ptr<InferenceResultFuture> Inference::infer(const rl::env::State &state)
    {
        for (int i = 0; i < 1000; i++)
        {
            try {
                auto batch = get_batch();
                auto label = batch->add_inference_request(state);
                return std::make_unique<InferenceResultFuture>(batch, label);
            } catch (InferenceBatchObsolete&) {
                continue;
            }
        }

        throw std::runtime_error{"Unable to find space in a batch for inference request."};
    }
}
