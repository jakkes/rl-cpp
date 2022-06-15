#ifndef RL_CPPUTILS_SLICE_VECTOR_H_
#define RL_CPPUTILS_SLICE_VECTOR_H_


#include <vector>


namespace rl::cpputils
{
    /**
     * @brief Slices a vector, in a copying manner.
     * 
     * @tparam T Type held by vector.
     * @param vec Vector.
     * @param start Start index.
     * @param end End index, exclusive. If smaller than zero, counts number of elements
     * to be excluded from the end of the vector.
     * @return std::vector<T> 
     */
    template<typename T>
    std::vector<T> slice(const std::vector<T> &vec, int start, int end)
    {
        if (end >= 0) end -= vec.size();
        assert(start >= 0);
        assert(vec.size() + end >= start);
        return std::vector<T>(vec.begin() + start, vec.end() + end);
    }
}

#endif /* RL_CPPUTILS_SLICE_VECTOR_H_ */
