#ifndef RL_ENV_BASE_H_
#define RL_ENV_BASE_H_


#include <memory>

#include <torch/torch.h>

#include "observation.h"
#include "state.h"


namespace rl::env
{

    /**
     * @brief Base class for all environments.
     * 
     */
    class Base{
        public:
            virtual std::unique_ptr<Observation> step(const torch::Tensor &action) = 0;
            virtual std::unique_ptr<State> reset() = 0;
            virtual std::unique_ptr<State> state() = 0;
            virtual bool is_terminal() = 0;

            void cuda() { is_cuda_ = true; }
            void cpu() { is_cuda_ = false; }
            inline bool is_cuda() { return is_cuda_; }
        
        private:
            bool is_cuda_{false};
    };

    class Factory {
        public:
            std::unique_ptr<Base> get() const;
            void cuda();
            void cpu();

        private:
            bool is_cuda{false};
            virtual std::unique_ptr<Base> get_impl() const = 0;
    };
}

#endif /* RL_ENV_BASE_H_ */
