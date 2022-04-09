#include "inference_result_future.h"


namespace rl::agents::ppo::trainers::seed_impl
{
    InferenceResultFuture::InferenceResultFuture(
        std::shared_ptr<InferenceBatch> batch,
        int64_t label
    ) :
    batch{batch}, label{label}
    {}
}
