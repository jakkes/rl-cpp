#include "rl/policies/categorical.h"

#include <stdexcept>
#include <memory>
#include <vector>

#include "rl/torchutils.h"
#include "rl/policies/constraints/categorical_mask.h"
#include "rl/policies/unsupported_constraint_exception.h"


using namespace torch::indexing;

namespace rl::policies
{
    Categorical::Categorical(const torch::Tensor &probabilities)
    : 
    probabilities{ register_buffer("probabilities", probabilities) },
    dim{probabilities.size(-1)},
    batch{probabilities.sizes().size() > 1}
    {
        check_probabilities();
        compute_internals();
    }
    
    void Categorical::compute_internals()
    {
        probabilities = probabilities / probabilities.sum(-1, true);

        if (named_buffers().contains("cumsummed")) {
            cumsummed.index_put_({Slice(None, None)}, probabilities.cumsum(-1));
        }
        else {
            cumsummed = register_buffer("cumsummed", probabilities.cumsum(-1));
        }

        sample_shape.reserve(cumsummed.sizes().size());
        sample_shape.clear();
        int i = 0;
        int64_t batchsize = 1;
        for (; i < cumsummed.sizes().size() - 1; i++) {
            batchsize *= cumsummed.size(i);
            sample_shape.push_back(cumsummed.size(i));
        }
        sample_shape.push_back(1);

        if (!named_buffers().contains("batchvec") && batch) {
            batchvec = register_buffer("batchvec", torch::arange(batchsize));
        }
    }

    void Categorical::check_probabilities()
    {
        if (probabilities.sizes().size() < 1) {
            throw std::invalid_argument{"Probabilities must have at least one dimension."};
        }
        if (probabilities.isnan().any().item().toBool()) {
            throw std::runtime_error{"Categorical policy did not expect NaN values."};
        }
        if (probabilities.lt(0.0).any().item().toBool()) {
            throw std::invalid_argument{"Probabilities must all be non-negative."};
        }
    }

    void Categorical::include(std::shared_ptr<constraints::Base> constraint)
    {
        auto categorical_mask = std::dynamic_pointer_cast<constraints::CategoricalMask>(constraint);

        if (categorical_mask) {
            auto mask = categorical_mask->mask().broadcast_to(probabilities.sizes());
            probabilities.index_put_({~mask}, 0.0);
            check_probabilities();
            compute_internals();
            return;
        }

        Base::include(constraint);
    }

    const torch::Tensor Categorical::get_probabilities() const
    {
        return probabilities;
    }

    torch::Tensor Categorical::sample() const
    {
        return (torch::rand(sample_shape, cumsummed.options()) > cumsummed).sum(-1).clamp_max_(dim-1);
    }

    torch::Tensor Categorical::entropy() const
    {
        auto logged = probabilities.log();
        logged.index_put_({logged.isinf()}, 0.0);
        return - (probabilities * logged).sum(-1);
    }

    torch::Tensor Categorical::log_prob(const torch::Tensor &value) const
    {
        return prob(value).log();
    }

    torch::Tensor Categorical::prob(const torch::Tensor &value) const
    {
        assert (rl::torchutils::is_int_dtype(value));

        if (batch) {
            return probabilities.view({-1, dim}).index({batchvec, value.view({-1})}).view(value.sizes());
        }
        else {
            assert(value.sizes().size() == 0);
            return probabilities.index({value});
        }
    }
}