#include "rl/agents/ppo/trainers/seed.h"

#include <memory>

#include <thread_safe/collections/queue.h>

#include "rl/buffers/tensor_and_object.h"
#include "rl/buffers/samplers/uniform.h"
#include "rl/cpputils/concat_vector.h"
#include "rl/cpputils/metronome.h"

#include "seed_impl/inference.h"
#include "seed_impl/actor.h"
#include "seed_impl/sequence.h"
#include "loss_fns.h"



using namespace rl;
using namespace torch::indexing;
using BufferType = buffers::TensorAndObject<std::shared_ptr<policies::constraints::Base>>;
using SamplerType = buffers::samplers::Uniform<BufferType>;
using DataStreamType = thread_safe::Queue<std::shared_ptr<agents::ppo::trainers::seed_impl::Sequence>>;

namespace rl::agents::ppo::trainers
{
    struct TensorInfo
    {
        std::vector<std::vector<int64_t>> shapes;
        std::vector<torch::Dtype> dtypes;
    };

    static TensorInfo get_tensor_shapes(
        std::shared_ptr<agents::ppo::Module> model,
        std::shared_ptr<env::Factory> env_factory,
        int64_t sequence_length,
        const SEEDOptions &options
    )
    {
        torch::NoGradGuard no_grad{};

        auto env = env_factory->get();
        auto state = env->reset();
        state->state = state->state.to(options.network_device);
        state->action_constraint->to(options.network_device);

        auto state_shape = state->state.sizes();

        auto policy = model->forward(state->state.unsqueeze(0));
        policy->policy->include(state->action_constraint);
        
        auto action = policy->policy->sample().squeeze(0);
        auto action_shape = action.sizes();

        TensorInfo out{};

        out.shapes = {
            cpputils::concat<int64_t>({sequence_length+1}, state_shape.vec()),
            cpputils::concat<int64_t>({sequence_length}, action_shape.vec()),
            std::vector<int64_t>{sequence_length},
            std::vector<int64_t>{sequence_length},
            std::vector<int64_t>{sequence_length},
            std::vector<int64_t>{sequence_length}
        };

        out.dtypes = {
            state->state.dtype().toScalarType(),
            action.dtype().toScalarType(),
            state->state.dtype().toScalarType(),
            torch::kBool,
            state->state.dtype().toScalarType(),
            state->state.dtype().toScalarType()
        };

        return out;
    }

    static std::vector<torch::TensorOptions> get_tensor_options(
        const std::vector<torch::Dtype> &dtypes,
        torch::Device device
    )
    {
        std::vector<torch::TensorOptions> out{};
        out.reserve(dtypes.size());

        for (const auto &dtype : dtypes)
        {
            out.push_back(
                torch::TensorOptions{}.dtype(dtype).device(device)
            );
        }
        return out;
    }


    class TrainerImpl
    {
        public:
            TrainerImpl(
                std::shared_ptr<rl::agents::ppo::Module> model,
                std::shared_ptr<torch::optim::Optimizer> optimizer,
                std::shared_ptr<rl::env::Factory> env_factory,
                const SEEDOptions &options
            ) :
            model{model},
            optimizer{optimizer},
            env_factory{env_factory},
            options{options}
            {
                auto tensor_info = get_tensor_shapes(model, env_factory, options.sequence_length, options);
                auto tensor_options = get_tensor_options(tensor_info.dtypes, options.replay_device);

                inference = std::make_shared<seed_impl::Inference>(
                    model,
                    seed_impl::InferenceOptions{}
                        .batchsize_(options.inference_batchsize)
                        .max_delay_ms_(options.inference_max_delay_ms)
                        .logger_(options.logger)
                        .device_(options.network_device)
                );
                inference_buffer = std::make_shared<BufferType>(
                    options.inference_replay_size,
                    tensor_info.shapes,
                    tensor_options
                );
                training_buffer = std::make_shared<BufferType>(
                    options.replay_size,
                    tensor_info.shapes,
                    tensor_options
                );
                training_sampler = std::make_shared<SamplerType>(
                    training_buffer
                );
                data_stream = std::make_shared<DataStreamType>(
                    options.inference_replay_size
                );
                actors.reserve(options.env_workers);
                for (int i = 0; i < options.env_workers; i++) {
                    actors.push_back(
                        std::make_shared<seed_impl::Actor>(
                            inference,
                            env_factory,
                            data_stream,
                            seed_impl::ActorOptions{}
                                .environments_(options.envs_per_worker)
                                .sequence_length_(options.sequence_length)
                                .logger_(options.logger)
                                .environment_device_(options.environment_device)
                        )
                    );
                }
            }

            ~TrainerImpl()
            {
                stop(); join();
            }

            void start()
            {
                if (running) return;

                running = true;
                for (auto &actor : actors) actor->start();

                inference_data_gathering_thread = std::thread(&TrainerImpl::inference_data_gatherer, this);
                training_thread = std::thread(&TrainerImpl::trainer, this);
            }

            void stop()
            {
                running = false;
                for (auto &actor : actors) actor->stop();
            }

            void join()
            {
                for (auto &actor : actors) actor->join();
                if (inference_data_gathering_thread.joinable()) inference_data_gathering_thread.join();
                if (training_thread.joinable()) training_thread.join();
            }

        private:
            std::atomic<bool> running{false};

            std::shared_ptr<rl::agents::ppo::Module> model;
            std::shared_ptr<torch::optim::Optimizer> optimizer;
            std::shared_ptr<rl::env::Factory> env_factory;
            SEEDOptions options;

            std::shared_ptr<seed_impl::Inference> inference;
            std::shared_ptr<DataStreamType> data_stream;
            std::shared_ptr<BufferType> inference_buffer;
            std::shared_ptr<BufferType> training_buffer;
            std::shared_ptr<SamplerType> training_sampler;
            std::vector<std::shared_ptr<seed_impl::Actor>> actors;

            std::mutex training_buffer_mtx{};

            std::thread inference_data_gathering_thread;
            std::thread training_thread;

            void inference_data_gatherer()
            {
                while (running)
                {
                    auto stream_result = data_stream->dequeue(std::chrono::seconds(1));
                    if (!stream_result) continue;

                    auto sequence = *stream_result;
                    std::shared_ptr<rl::policies::constraints::Base> constraints = policies::constraints::stack(sequence->constraints);
                    constraints->to(options.replay_device);
                    inference_buffer->add(
                        {
                            torch::stack(sequence->states, 0).unsqueeze_(0).to(options.replay_device),
                            torch::stack(sequence->actions, 0).unsqueeze_(0).to(options.replay_device),
                            torch::tensor(sequence->rewards, torch::TensorOptions{}.dtype(sequence->states[0].dtype().toScalarType()).device(options.replay_device)).unsqueeze_(0).to(options.replay_device),
                            torch::tensor(sequence->not_terminals, torch::TensorOptions{}.dtype(torch::kBool).device(options.replay_device)).unsqueeze_(0).to(options.replay_device),
                            torch::stack(sequence->action_probabilities, 0).unsqueeze_(0).to(options.replay_device),
                            torch::stack(sequence->state_values, 0).unsqueeze_(0).to(options.replay_device)
                        },
                        {constraints}
                    );

                    if (inference_buffer->size() == options.inference_replay_size)
                    {
                        auto data = inference_buffer->get_all();

                        std::lock_guard lock{training_buffer_mtx};
                        training_buffer->add(
                            data->tensors, data->objs
                        );

                        inference_buffer->clear();
                    }
                }
            }

            void trainer()
            {
                while (running && training_buffer->size() < options.min_replay_size) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }

                size_t period{0};
                if (options.max_update_frequency > 0) {
                    period = static_cast<size_t>(1.0 / options.max_update_frequency * 1e9);
                }
                rl::cpputils::Metronome<std::chrono::nanoseconds> metronome{period};

                while (running)
                {
                    metronome.spin();
                    std::lock_guard lock{training_buffer_mtx};
                    auto sample = training_sampler->sample(options.batchsize);
                    auto stacked_constraints = policies::constraints::stack(sample->objs);
                    stacked_constraints->to(options.network_device);
                    for (auto &x : sample->tensors) {
                        x = x.to(options.network_device);
                    }

                    auto model_output = model->forward(sample->tensors[0].index({Slice(), Slice(None, -1)}));
                    model_output->policy->include(stacked_constraints->index({Slice(), Slice(None, -1)}));
                    auto action_probabilities = model_output->policy->prob(sample->tensors[1]);

                    auto last_state_output = model->forward(sample->tensors[0].index({Slice(), Slice(-1, None)}));
                    auto values = torch::cat({model_output->value, last_state_output->value}, 1);

                    auto deltas = compute_deltas(sample->tensors[2], values, sample->tensors[3], options.discount);
                    auto advantages = compute_advantages(deltas.detach(), sample->tensors[3], options.discount, options.gae_discount);
                    auto value_loss = compute_value_loss(deltas);
                    auto policy_loss = compute_policy_loss(advantages, sample->tensors[4], action_probabilities, options.eps);

                    auto loss = value_loss + policy_loss;

                    if (loss.isnan().any().item().toBool()) {
                        std::cout << "NaN loss!\n";
                    }

                    optimizer->zero_grad();
                    loss.backward();
                    optimizer->step();

                    if (options.logger) {
                        options.logger->log_scalar("Trainer/ValueLoss", value_loss.item().toFloat());
                        options.logger->log_scalar("Trainer/PolicyLoss", policy_loss.item().toFloat());
                        options.logger->log_frequency("Trainer/UpdateFrequency", 1);
                    }
                }
            }
    };


    SEED::SEED(
        std::shared_ptr<rl::agents::ppo::Module> model,
        std::shared_ptr<torch::optim::Optimizer> optimizer,
        std::shared_ptr<rl::env::Factory> env_factory,
        const SEEDOptions &options
    ) :
    model{model}, optimizer{optimizer}, env_factory{env_factory}, options{options}
    {}

    template<class Rep, class Period>
    void SEED::run(std::chrono::duration<Rep, Period> duration)
    {
        auto data_stream = std::make_shared<thread_safe::Queue<std::shared_ptr<seed_impl::Sequence>>>(options.inference_replay_size);

        auto start = std::chrono::steady_clock::now();
        auto end = start + duration;

        TrainerImpl t{model, optimizer, env_factory, options};
        t.start();

        while (std::chrono::steady_clock::now() < end) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        t.stop();
        t.join();
    }

    template void SEED::run<int64_t, std::ratio<1L>>(std::chrono::duration<int64_t, std::ratio<1L>> duration);
}
