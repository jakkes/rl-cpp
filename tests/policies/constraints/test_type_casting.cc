#include <torch/torch.h>
#include <gtest/gtest.h>

#include <rl/policies/constraints/constraints.h>


using namespace rl::policies::constraints;

TEST(constraints, test_is_type)
{
    std::shared_ptr<Base> constraint = std::make_shared<Box>(torch::zeros({5}), torch::ones({5}));

    ASSERT_TRUE(constraint->is_type<Box>());
    ASSERT_FALSE(constraint->is_type<CategoricalMask>());
}

TEST(constraints, test_is_type_pure_ptr)
{
    Base *constraint = new Box{torch::zeros({5}), torch::ones({5})};

    ASSERT_TRUE(constraint->is_type<Box>());
    ASSERT_FALSE(constraint->is_type<CategoricalMask>());

    delete constraint;
}

TEST(constraints, test_as_type)
{
    std::shared_ptr<Base> constraint = std::make_shared<Box>(torch::zeros({5}), torch::ones({5}));

    auto &box = constraint->as_type<Box>();
    ASSERT_ANY_THROW(constraint->as_type<CategoricalMask>());
}

TEST(constraints, test_as_type_pure_ptr)
{
    Base *constraint = new Box{torch::zeros({5}), torch::ones({5})};

    auto &box = constraint->as_type<Box>();
    ASSERT_ANY_THROW(constraint->as_type<CategoricalMask>());

    delete constraint;
}
