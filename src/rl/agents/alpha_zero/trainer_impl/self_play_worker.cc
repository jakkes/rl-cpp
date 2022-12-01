#include "self_play_worker.h"

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

    void SelfPlayWorker::worker()
    {
        while (running) {
            
        }
    }
}
