#include "self_play_worker.h"


using namespace torch::indexing;

namespace trainer_impl
{
    SelfPlayWorker::SelfPlayWorker(
        std::shared_ptr<rl::simulators::Base> simulator,
        std::shared_ptr<modules::Base> module,
        const SelfPlayWorkerOptions &options
    ) : simulator{simulator}, module{module}, options{options}
    {}

    void SelfPlayWorker::start()
    {
        running = true;
        working_thread = std::thread(&SelfPlayWorker::worker, this);
    }

    void SelfPlayWorker::stop()
    {
        running = false;
        if (working_thread.joinable()) {
            working_thread.join();
        }
    }

    void SelfPlayWorker::set_initial_state()
    {
        auto initial_states = simulator->reset(options.batchsize);
        states = initial_states.states.unsqueeze(1);
        masks = std::dynamic_pointer_cast<rl::policies::constraints::CategoricalMask>(
            initial_states.action_constraints
        )->mask().unsqueeze(1);
        rewards = torch::zeros({options.batchsize, 0});
    }

    void SelfPlayWorker::step()
    {
        auto root_nodes = mcts(
            states.index({Slice(), -1}),
            masks.index({Slice(), -1}),
            module,
            simulator,
            options.mcts_options
        );
    }

    void SelfPlayWorker::worker()
    {
        set_initial_state();

        while (running) {
            step();
        }
    }
}
