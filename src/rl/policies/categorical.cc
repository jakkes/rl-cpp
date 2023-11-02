#include "rl/policies/categorical.h"

#include <stdexcept>
#include <memory>
#include <vector>

#include <rl/torchutils/torchutils.h>
#include "rl/policies/constraints/categorical_mask.h"
#include "rl/policies/unsupported_constraint_exception.h"
#include "rl/cpputils/slice_vector.h"
#include "rl/cpputils/concat_vector.h"


using namespace torch::indexing;

namespace rl::policies
{
    Categorical::Categorical(const torch::Tensor &probabilities, const torch::Tensor &values)
    :
        probabilities{probabilities},
        values{values},
        dim{probabilities.size(-1)},
        sample_shape{rl::cpputils::slice(probabilities.sizes().vec(), 0, -1)}
    {
        compute_internals();
    }

    Categorical::Categorical(const torch::Tensor &probabilities)
    : Categorical{
        probabilities,
        torch::arange(probabilities.size(-1), torch::TensorOptions{}.device(probabilities.device())).expand_as(probabilities)
    }
    {}

    void Categorical::compute_internals()
    {
        probabilities = probabilities / probabilities.sum(-1, true);
        probabilities = probabilities.view({-1, dim});
        values = values.view_as(probabilities);

        cumsummed = probabilities.cumsum(1);
        batchvec = torch::arange(probabilities.size(0), torch::TensorOptions{}.device(probabilities.device()));
    }

    void Categorical::include(std::shared_ptr<constraints::Base> constraint)
    {
        auto categorical_mask = std::dynamic_pointer_cast<constraints::CategoricalMask>(constraint);

        if (categorical_mask) {
            auto mask = categorical_mask->mask().view_as(probabilities);
            probabilities.index_put_({~mask}, 0.0);
            compute_internals();
            return;
        }

        Base::include(constraint);
    }

    const torch::Tensor Categorical::get_probabilities() const
    {
        return probabilities.view(rl::cpputils::concat(sample_shape, {dim}));
    }

    torch::Tensor Categorical::sample() const
    {
        auto actions = (torch::rand(sample_shape, cumsummed.options()).unsqueeze_(1) > cumsummed).sum(1);
        actions.clamp_max_(dim - 1);
        auto out = values.index({batchvec, actions});
        return out.view(sample_shape);
    }

    torch::Tensor Categorical::entropy() const
    {
        auto logged = probabilities.log();
        logged.index_put_({logged.isinf()}, 0.0);
        return - (probabilities * logged).sum(-1).view(sample_shape);
    }

    torch::Tensor Categorical::log_prob(const torch::Tensor &value) const
    {
        return prob(value).log();
    }

    torch::Tensor Categorical::prob(const torch::Tensor &value_) const
    {
        auto value = value_.view({-1, 1});
        auto i = torch::where(value == values);
        assert (i.size() == 2);

        return probabilities.index({i[0], i[1]}).view_as(value_);
    }
}