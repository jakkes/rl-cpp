#include <gtest/gtest.h>

#include <rl/env/base.h>


class Simple : public rl::env::Base
{
    public:
        std::unique_ptr<const rl::env::Observation> step(torch::Tensor action)
        {
            auto observation = std::make_unique<const rl::env::Observation>();
            observation->state->state = torch::tensor(x);

            if (x == 10) {

            }

            return observation;
        }
        
    private:
        int x{0};
        int limit{10};
};


TEST(test_env, test_simple_env)
{
    ASSERT_EQ(1, 1);
}
