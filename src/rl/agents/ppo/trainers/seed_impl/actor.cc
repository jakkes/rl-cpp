#include "actor.h"

#include <algorithm>


namespace rl::agents::ppo::trainers::seed_impl
{

    Actor::Actor(
        std::shared_ptr<Inference> inference,
        std::shared_ptr<rl::env::Factory> env_factory,
        std::shared_ptr<thread_safe::Queue<std::shared_ptr<Sequence>>> out_stream,
        const ActorOptions &options
    ) :
    inference{inference}, env_factory{env_factory}, out_stream{out_stream}, options{options}
    {}

    void Actor::start()
    {
        if (is_running) throw std::runtime_error{"Actor already started."};
        is_running = true;
        working_thread = std::thread(&Actor::worker, this);
    }

    void Actor::stop()
    {
        is_running = false;
    }

    void Actor::join()
    {
        stop();
        if (working_thread.joinable()) working_thread.join();
    }

    struct Env
    {
        std::shared_ptr<rl::env::Base> env;
        std::unique_ptr<InferenceResultFuture> inference_result;
        std::shared_ptr<Sequence> sequence;
        int step;
    };

    void Actor::worker()
    {
        std::vector<Env> envs{};
        envs.reserve(options.environments);

        for (int i = 0; i < options.environments; i++) {
            envs.push_back(
                {env_factory->get(), nullptr, nullptr, 0}
            );
        }

        for (int i = 0; i < options.environments; i++) {

        }

        int i = 0;
        while (is_running)
        {
            i = (i + 1) % options.environments;
        }
    }
}
