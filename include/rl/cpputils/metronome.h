#ifndef RL_CPPUTILS_METRONOME_H_
#define RL_CPPUTILS_METRONOME_H_


#include <chrono>
#include <thread>
#include <functional>
#include <atomic>


namespace rl::cpputils
{
    class Metronome
    {
        public:
            Metronome(std::function<void()> callback);

            ~Metronome();

            void stop();

            template<typename Rep, typename Period>
            void start(std::chrono::duration<Rep, Period> period) {
                _is_running = true;
                working_thread = std::thread(&Metronome::worker<Rep, Period>, this, period);
            }

            inline
            bool is_running() { return _is_running; }

        private:
            std::function<void()> callback;
            std::atomic<bool> _is_running{false};
            std::thread working_thread;

            template<typename Rep, typename Period>
            void worker(std::chrono::duration<Rep, Period> period)
            {
                auto last_call = std::chrono::high_resolution_clock::now();
                while (_is_running)
                {
                    auto next_call = last_call + period;
                    std::this_thread::sleep_until(next_call);
                    callback();
                    last_call = next_call;
                }
            }
    };
}

#endif /* RL_CPPUTILS_METRONOME_H_ */
