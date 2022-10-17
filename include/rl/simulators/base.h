#ifndef RL_SIMULATORS_BASE_H_
#define RL_SIMULATORS_BASE_H_


#include <memory>

#include <torch/torch.h>

#include <rl/policies/base.h>
#include <rl/logging/client/base.h>

namespace rl::simulators
{
    struct State
    {
        torch::Tensor state;
        std::shared_ptr<rl::policies::Base> action_constraint;
    };

    struct Transition
    {
        State state;
        torch::Tensor action;
        float reward;
        bool terminal;
        State next_state;
    };

    class Base
    {
        public:
            virtual ~Base() = default;

            virtual
            State reset() const = 0;

            virtual
            Transition step(const torch::Tensor &state, const torch::Tensor &action) const = 0;

            inline
            void set_logger(std::shared_ptr<rl::logging::client::Base> logger) {
                this->logger = logger;
            }
        
        protected:
            std::shared_ptr<rl::logging::client::Base> logger;
    };

    class Factory
    {
        public:
            virtual std::unique_ptr<Base> get() const = 0;
    };
}

#endif /* RL_SIMULATORS_BASE_H_ */
