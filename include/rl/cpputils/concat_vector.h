#ifndef RL_CPPUTILS_CONCAT_VECTOR_H_
#define RL_CPPUTILS_CONCAT_VECTOR_H_


#include <vector>


namespace rl::cpputils
{
    template<typename T>
    std::vector<T> concat(const std::vector<T> &a, const std::vector<T> &b)
    {
        std::vector<T> out{};
        out.reserve(a.size() + b.size());

        for (const auto &x : a) out.push_back(x);
        for (const auto &x : b) out.push_back(x);

        return out;
    }
}

#endif /* RL_CPPUTILS_CONCAT_VECTOR_H_ */
