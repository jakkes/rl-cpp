#include "env_thread.h"


namespace seed_impl
{
    EnvThread::EnvThread(
        std::shared_ptr<rl::env::Factory> env_factory,
        std::shared_ptr<Inferer> inferer,
        const rl::agents::dqn::trainers::SEEDOptions &options
    ) : options{options}
    {
        env_factory = env_factory;
        inferer = inferer;
    }

    void EnvThread::start() {
        running = true;
        worker_thread = std::thread(&EnvThread::worker, this);
    }

    void EnvThread::stop() {
        running = false;
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }

    void EnvThread::worker()
    {
        workers.reserve(options.envs_per_worker);
        for (int i = 0; i < options.envs_per_worker; i++) {
            workers.emplace_back(env_factory, inferer, options);
        }

        while (running)
        {

        }
    }

    EnvWorker::EnvWorker(
        std::shared_ptr<rl::env::Factory> env_factory,
        std::shared_ptr<Inferer> inferer,
        const rl::agents::dqn::trainers::SEEDOptions &options
    ) : options{options}
    {
        this->inferer = inferer;
        env = env_factory->get();
    }
}
