#include <string>
#include <atomic>
#include <unordered_map>

#include <gtest/gtest.h>
#include <rl/rl.h>


using namespace rl::logging;

TEST(test_logging, test_unordered_map_inplace_change)
{
    std::unordered_map<std::string, std::atomic<size_t>> map{};
    map["abc"] = 10;

    auto start = map.begin();
    start->second = 5;

    ASSERT_EQ(map["abc"], 5);
}

TEST(test_logging, test_ema)
{
    client::EMA logger{{0.6, 0.9, 0.99}, 1};

    for (int i = 0; i < 100; i++) {
        logger.log_scalar("step", i);
        logger.log_frequency("stepfreq", 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}
