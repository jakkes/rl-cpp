#ifndef RL_CPPUTILS_CONCAT_VECTOR_H_
#define RL_CPPUTILS_CONCAT_VECTOR_H_


#include <vector>


namespace rl::cpputils
{
    /**
     * @brief Concatenates two vectors.
     * 
     * @tparam T object type held by vectors.
     * @param a First part of output vector.
     * @param b Second part of output vector.
     * @return std::vector<T> Vector of `a` followed by `b`.
     */
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
