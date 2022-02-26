#ifndef RL_CPPUTILS_H_
#define RL_CPPUTILS_H_


#include <vector>


namespace rl::cpputils
{
    template<typename T>
    std::vector<T> slice(const std::vector<T> &vec, int start, int end)
    {
        if (end >= 0) end -= vec.size();
        return std::vector<T>(vec.begin() + start, vec.end() + end);
    }
}

#endif /* RL_CPPUTILS_H_ */
