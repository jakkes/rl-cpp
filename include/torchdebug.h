#ifndef RL_TORCHDEBUG_H_
#define RL_TORCHDEBUG_H_


#include <iostream>

#include <torch/torch.h>


void print(torch::Tensor &x) {
    std::cout << x << "\n";
}
void print(torch::Tensor &&x) { print(x); }

#endif /* RL_TORCHDEBUG_H_ */
