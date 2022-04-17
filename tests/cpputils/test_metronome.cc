#include <atomic>

#include <gtest/gtest.h>

#include "rl/cpputils/metronome.h"


using namespace rl::cpputils;


TEST(test_cpputils, test_metronome)
{
    std::atomic<size_t> count{0};

    auto increment = [&] () { count++; };

    Metronome<std::chrono::milliseconds> m{100};
    m.start(increment);

    std::this_thread::sleep_for(std::chrono::seconds(2));
    m.stop();

    ASSERT_TRUE(count == 20 || count == 21);
}
