#include "inference.h"


namespace rl::agents::ppo::trainers::seed_impl
{



    Inference::Inference(
        std::shared_ptr<rl::agents::ppo::Module> model,
        const InferenceOptions &options
    ) :
    model{model}, options{options}
    {}


}
