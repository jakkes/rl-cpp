#include "transition_collector.h"
#include "helpers.h"


namespace seed_impl
{
    TransitionCollector::TransitionCollector(
        std::shared_ptr<thread_safe::Queue<rl::utils::NStepCollectorTransition>> transition_queue,
        std::shared_ptr<rl::buffers::Tensor> training_buffer,
        std::shared_ptr<rl::env::Factory> env_factory,
        const rl::agents::dqn::trainers::SEEDOptions &options
    ) : transition_queue{transition_queue}, training_buffer{training_buffer}, options{options}
    {
        inference_buffer = create_buffer(options.inference_replay_size, env_factory, options);
    }

    void TransitionCollector::start()
    {
        running = true;
        working_thread = std::thread(&TransitionCollector::worker, this);
    }

    void TransitionCollector::stop()
    {
        running = false;
        if (working_thread.joinable()) {
            working_thread.join();
        }
    }

    void TransitionCollector::worker()
    {
        auto all_indices = torch::arange(options.inference_replay_size);

        while (running)
        {
            auto transition_ptr = transition_queue->dequeue(std::chrono::milliseconds(500));
            if (!transition_ptr) {
                continue;
            }

            auto &transition = *transition_ptr;
            inference_buffer->add({
                transition.state->state.unsqueeze(0).to(options.replay_device),
                get_mask(*transition.state->action_constraint).unsqueeze(0).to(options.replay_device),
                transition.action.unsqueeze(0).to(options.replay_device),
                torch::tensor({transition.reward}, inference_buffer->tensor_options()[3]),
                torch::tensor({!transition.terminal}, inference_buffer->tensor_options()[4]),
                transition.next_state->state.unsqueeze(0).to(options.replay_device),
                get_mask(*transition.next_state->action_constraint).unsqueeze(0).to(options.replay_device)
            });

            if (inference_buffer->size() == options.inference_replay_size) {
                auto data = inference_buffer->get(all_indices);
                training_buffer->add(*data);
                inference_buffer->clear();
            }
        }
    }
}
