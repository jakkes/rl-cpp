#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torchdebug.h>
#include <torch_test.h>

#include <rl/torchutils/torchutils.h>


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
