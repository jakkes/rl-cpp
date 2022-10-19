#include "inferer.h"


namespace seed_impl
{
    Inferer::Inferer(
        std::shared_ptr<rl::agents::dqn::modules::Base> module,
        std::shared_ptr<rl::agents::dqn::policies::Base> policy,
        const rl::agents::dqn::trainers::SEEDOptions &options
    ) : options{options}
    {
        this->module = module;
        this->policy = policy;
        batch = std::make_shared<InferenceBatch>(module, policy, &this->options);
    }

    std::unique_ptr<InferenceResultFuture> Inferer::infer(
        const torch::Tensor &state,
        const torch::Tensor &mask
    ) {
        std::lock_guard lock{infer_mtx};

        auto id = batch->try_add(state, mask);
        if (id < 0) {
            batch = std::make_shared<InferenceBatch>(module, policy, &options);
            id = batch->try_add(state, mask);
        }
        assert (id >= 0);

        return std::make_unique<InferenceResultFuture>(batch, id);
    }

    InferenceBatch::InferenceBatch(
        std::shared_ptr<rl::agents::dqn::modules::Base> module,
        std::shared_ptr<rl::agents::dqn::policies::Base> policy,
        const rl::agents::dqn::trainers::SEEDOptions *options
    ) : options{options}, module{module}, policy{policy} {
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
        std::lock_guard lock{execute_mtx};
        if (full()) {
            return -1;
        }

        if (empty()) {
            start_worker();
        }

        states.push_back(state);
        masks.push_back(mask);

        add_cv.notify_all();

        return size() - 1;
    }

    void InferenceBatch::start_worker() {
        worker_thread = std::thread(&InferenceBatch::worker, this);
    }

    void InferenceBatch::worker() {
        auto start = std::chrono::high_resolution_clock::now();
        auto end = start + std::chrono::milliseconds(options->inference_max_delay_ms);
        
        std::unique_lock lock{execute_mtx};
        add_cv.wait_until(lock, end, [&] () { return full(); });

        assert(lock.owns_lock());
        execute();

        if (options->logger) {
            options->logger->log_scalar("SEEDDQN/Inference batch size", states.size());
            options->logger->log_scalar("SEEDDQN/Inference delay (ms)", (end-start).count() / 1e6);
        }
    }

    void InferenceBatch::execute() {
        torch::InferenceMode guard{};
        auto output = module->forward(torch::stack(states));
        output->apply_mask(torch::stack(masks));

        value = output->value();
        actions = policy->policy(*output)->sample();
        executed_ = true;
        executed_cv.notify_all();
    }

    std::unique_ptr<InferenceResult> InferenceBatch::get(int64_t id)
    {
        if (!executed()) {
            std::mutex mtx{};
            std::unique_lock lock{mtx};
            executed_cv.wait(lock, [&]() { return executed(); });
        }

        auto out = std::make_unique<InferenceResult>();
        out->action = actions.index({id});
        out->value = value.index({id});
        return out;
    }
}
