#include "rl/policies/categorical.h"

#include <stdexcept>
#include <vector>

#include "rl/torchutils.h"


namespace rl::policies
{
    static std::set<torch::ScalarType> allowed_dtypes{{ torch::kInt64, torch::kInt32, torch::kInt16, torch::kInt8 }};

    Categorical::Categorical(const torch::Tensor &probabilities)
    : probabilities{probabilities}
    {
        if (probabilities.sizes().size() > 2) {
            throw std::invalid_argument{"Probabilities can only have one batch dimension."};
        }
        if (probabilities.sizes().size() < 1) {
            throw std::invalid_argument{"Probabilities must have at least one dimension."};
        }
        if (probabilities.isnan().any().item().toBool()) {
            throw std::runtime_error{"Categorical policy did not expect NaN values."};
        }
        if (probabilities.lt(0.0).any().item().toBool()) {
            throw std::invalid_argument{"Probabilities must all be non-negative."};
        }

        this->probabilities /= probabilities.sum(-1, true);
        assert(this->probabilities.sum(-1).eq(1.0).all().item().toBool());

        cumsummed = this->probabilities.cumsum(-1);
        log_probabilities = this->probabilities.log();

        batch = cumsummed.sizes().size() == 2;
        batchsize = cumsummed.size(0);
        batchvec = torch::arange(batchsize);
    }

    const torch::Tensor Categorical::get_probabilities() const
    {
        return probabilities;
    }

    const torch::Tensor Categorical::sample() const
    {
        auto sample = batch ? torch::rand({batchsize, 1}, cumsummed.options()) : torch::rand({1}, cumsummed.options());
        return (sample > cumsummed).sum(-1);
    }

    const torch::Tensor Categorical::log_prob(const torch::Tensor &value) const
    {
        assert (rl::torchutils::is_int_dtype(value));
        
        if (batch)
        {
            assert(value.sizes().size() == 1);
            return log_probabilities.index({batchvec, value});
        }
        else
        {
            assert(value.sizes().size() == 0);
            return log_probabilities.index({value});
        }
    }
}