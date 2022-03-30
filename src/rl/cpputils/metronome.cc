#include "rl/cpputils/metronome.h"


namespace rl::cpputils
{
    Metronome::Metronome(std::function<void()> callback)
    : callback{callback}
    {}

    Metronome::~Metronome() {
        stop();
    }

    void Metronome::stop() {
        _is_running = false;
        if (working_thread.joinable()) {
            working_thread.join();
        }
    }
}
