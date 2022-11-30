#include "inferer.h"

#include <rl/agents/alpha_zero/mcts.h>


namespace seed_impl
{
    Inferer::Inferer(
        std::shared_ptr<modules::Base> module,
        std::shared_ptr<rl::simulators::Base> simulator,
        const trainers::SEEDOptions &options
    ) : options{options}
    {
        this->module = module;
        this->simulator = simulator;
    }

    void Inferer::new_batch()
    {
        batches.push_back(
            std::make_shared<InferenceBatch>(
                module,
                simulator,
                std::bind(&Inferer::batch_stale_callback, this, std::placeholders::_1),
                &options
            )
        );
    }

    std::unique_ptr<InferenceResultFuture> Inferer::infer(
        const torch::Tensor &state,
        const torch::Tensor &mask
    ) {
        std::lock_guard lock{infer_mtx};

        if (batches.size() == 0) {
            new_batch();
        }
        auto id = batches.back()->try_add(state, mask);
        if (id < 0) {
            new_batch();
            id = batches.back()->try_add(state, mask);
        }
        assert (id >= 0);

        return std::make_unique<InferenceResultFuture>(batches.back(), id);
    }

    void Inferer::batch_stale_callback(InferenceBatch *batch)
    {
        assert(batch == batches.front().get());
        std::lock_guard lock{infer_mtx};
        batch_queue.enqueue(batches.front());
        batches.pop_front();
    }

    void Inferer::start()
    {
        running = true;
        worker_thread = std::thread(&Inferer::worker, this);
    }

    void Inferer::stop()
    {
        running = false;
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }

    void Inferer::worker()
    {
        while (running)
        {
            auto batch_ptr = batch_queue.dequeue(std::chrono::milliseconds(500));
            if (!batch_ptr) {
                continue;
            }

            auto batch = *batch_ptr;
            batch->execute();
        }
    }

    InferenceBatch::InferenceBatch(
        std::shared_ptr<modules::Base> module,
        std::shared_ptr<rl::simulators::Base> simulator,
        std::function<void(InferenceBatch*)> stale_callback,
        const trainers::SEEDOptions *options
    ) : options{options}, module{module},
        simulator{simulator}, stale_callback{stale_callback}
    {
        states.reserve(options->inference_batchsize);
        masks.reserve(options->inference_batchsize);
    }

    InferenceBatch::~InferenceBatch()
    {
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }

    int64_t InferenceBatch::try_add(const torch::Tensor &state, const torch::Tensor &mask)
    {
        std::lock_guard lock{add_mtx};
        if (full() || stale()) {
            return -1;
        }

        if (empty()) {
            start_worker();
        }

        states.push_back(state);
        masks.push_back(mask);

        add_cv.notify_one();

        return size() - 1;
    }

    void InferenceBatch::start_worker() {
        worker_thread = std::thread(&InferenceBatch::worker, this);
    }

    void InferenceBatch::worker() {
        start_time = std::chrono::high_resolution_clock::now();
        auto end = start_time + std::chrono::milliseconds(options->inference_max_delay_ms);
        
        {
            std::unique_lock lock{add_mtx};
            add_cv.wait_until(lock, end, [&] () { return full(); });
            stale_ = true;   
        }
        stale_callback(this);
    }


    void InferenceBatch::execute() {
        torch::InferenceMode guard{};
        std::lock_guard lock{add_mtx};
        assert(stale() && !executed());

        result_nodes = mcts(
            torch::stack(states, 0),
            std::make_shared<rl::policies::constraints::CategoricalMask>(torch::stack(masks, 0)),
            module,
            simulator,
            MCTSOptions{}
                .module_device_(options->inference_device)
                .steps_(options->inference_mcts_steps)
        );

        executed_ = true;
        executed_cv.notify_all();

        if (options->logger) {
            options->logger->log_scalar("SEEDDQN/Inference batch size", states.size());
            options->logger->log_frequency("SEEDDQN/Step frequency", states.size());
            options->logger->log_scalar("SEEDDQN/Inference delay", (std::chrono::high_resolution_clock::now() - start_time).count() / 1e6);
        }
    }

    std::unique_ptr<InferenceResult> InferenceBatch::get(int64_t id)
    {
        if (!executed()) {
            std::mutex mtx{};
            std::unique_lock lock{mtx};

            bool did_execute{false};
            while (!did_execute) {
                did_execute = executed_cv.wait_for(
                    lock,
                    std::chrono::seconds(1),
                    [&]() { return executed();
                });
            }
        }

        auto out = std::make_unique<InferenceResult>();
        out->action = actions.index({id});
        out->value = value.index({id});
        out->advantage = advantage.index({id});
        return out;
    }
}