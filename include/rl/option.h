#ifndef RL_OPTION_H_
#define RL_OPTION_H_


#define RL_OPTION(TypeName, ValueName) \
auto &ValueName##_(const TypeName &value) \
{ \
    this->ValueName = value; \
    return *this; \
} \
TypeName ValueName


#endif /* RL_OPTION_H_ */
