add_executable(alpha_zero_combinatorial_lock combinatorial_lock.cc)
target_link_libraries(alpha_zero_combinatorial_lock PRIVATE rl::rl argparse)

add_executable(alpha_zero_cart_pole cart_pole.cc)
target_link_libraries(alpha_zero_cart_pole PRIVATE rl::rl argparse)
