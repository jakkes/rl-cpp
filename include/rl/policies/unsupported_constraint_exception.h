#ifndef RL_POLICIES_UNSUPPORTED_CONSTRAINT_EXCEPTION_H_
#define RL_POLICIES_UNSUPPORTED_CONSTRAINT_EXCEPTION_H_


#include <stdexcept>

namespace rl::policies
{
    class UnsupportedConstraintException : public std::exception {};
}


#endif /* RL_POLICIES_UNSUPPORTED_CONSTRAINT_EXCEPTION_H_ */
