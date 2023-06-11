#ifndef RL_ENV_BASE_H_
#define RL_ENV_BASE_H_


#include <memory>
#include <vector>
#include <mutex>

#include <torch/torch.h>

#include "rl/logging/client/base.h"

#include "observation.h"
#include "state.h"


namespace rl::env
{
    /**
     * @brief Environment base class.
     * 
     * An environment is a stateful system on which one may apply actions to affect the
     * state. The quality of an action, in the given state, is determined by a reward,
     * returned after every applied action.
     * 
     */
    class Base{
        public:

            virtual ~Base() = default;

            /**
             * @brief Applies an action to the environment.
             * 
             * @param action Action, of some appropriate format (as defined by the
             *  environment).
             * @return std::unique_ptr<Observation> Observation of the state transition.
             */
            virtual std::unique_ptr<Observation> step(const torch::Tensor &action) = 0;
            
            /**
             * @brief Resets the environment to some initial state.
             * 
             * @return std::unique_ptr<State> Initial state.
             */
            virtual std::unique_ptr<State> reset() = 0;
            
            /**
             * @brief Returns the current state of the environment.
             * 
             * @return std::unique_ptr<State> Current state.
             */
            virtual std::unique_ptr<State> state() const = 0;
            
            /**
             * @brief Whether or not the environment is in a terminal state. If so, it
             *  should be reset, using `reset`.
             * 
             * @return true Environment is in a terminal state.
             * @return false Environment is not in a terminal state.
             */
            virtual bool is_terminal() const = 0;

            /**
             * @brief Set the logger object used by the environment.
             * 
             * @param logger 
             */
            void set_logger(std::shared_ptr<rl::logging::client::Base> logger);

            /**
             * @brief Indicate that the environment should interface using CUDA tensors.
             * 
             */
            void cuda();

            /**
             * @brief Indicate that the environment should interface using CPU tensors.
             * 
             */
            void cpu();

            /**
             * @brief Whether or not the environment is interfacing CUDA or CPU tensors.
             * 
             * @return true CUDA tensors are used.
             * @return false CPU tensors are used.
             */
            inline bool is_cuda() const { return is_cuda_; }
        
        protected:
            std::shared_ptr<rl::logging::client::Base> logger;
            bool is_cuda_{false};
    };

    /**
     * @brief Base class for environment factories. An environment factory spawns new
     * environment instances on request.
     * 
     */
    class Factory {
        public:

            virtual ~Factory() = default;

            /**
             * @brief Spawns a new environment instance.
             * 
             * @return std::unique_ptr<Base> Environment pointer.
             */
            std::unique_ptr<Base> get() const;

            /**
             * @brief Set the logger object used by spawned environment.
             * 
             * @param logger 
             */
            void set_logger(std::shared_ptr<rl::logging::client::Base> logger);

            /**
             * @brief Future environment instances spawned will interface using CUDA
             * backed tensors.
             * 
             */
            void cuda();

            /**
             * @brief Future environment instances spawned will interface using CPU
             * backed tensors.
             * 
             */
            void cpu();

        private:
            bool is_cuda{false};
            std::shared_ptr<rl::logging::client::Base> logger;
            virtual std::unique_ptr<Base> get_impl() const = 0;
    };
}

#endif /* RL_ENV_BASE_H_ */
