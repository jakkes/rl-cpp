#ifndef INCLUDE_RL_BUFFERS_SAMPLERS_UNIFORM_H_
#define INCLUDE_RL_BUFFERS_SAMPLERS_UNIFORM_H_


#include <memory>

#include <torch/torch.h>


namespace rl::buffers::samplers
{
    template<typename T>
    class Uniform{
        public:
            Uniform(std::shared_ptr<T> buffer) : buffer{buffer} {}

            auto sample(int64_t n)
            {
                return buffer->get(
                    torch::randint(buffer->size(), {n}, torch::TensorOptions{}.dtype(torch::kLong))
                );
            }
        
        private:
            std::shared_ptr<T> buffer;
    };
}

#endif /* INCLUDE_RL_BUFFERS_SAMPLERS_UNIFORM_H_ */
