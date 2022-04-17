#ifndef RL_CPPUTILS_METRONOME_H_
#define RL_CPPUTILS_METRONOME_H_


#include <chrono>
#include <thread>
#include <functional>
#include <atomic>


namespace rl::cpputils
{
    template<typename PeriodType = std::chrono::seconds>
    class Metronome
    {
        public:
            Metronome(size_t period)
            : period{PeriodType(period)} {}

            ~Metronome() {
                stop();
            }

            void stop() {
                _is_running = false;
                if (working_thread.joinable()) {
                    working_thread.join();
                }
            }

            void start(std::function<void()> callback) {
                if (_is_running) throw std::runtime_error{"Metronome already in a running state."};
                _is_running = true;
                working_thread = std::thread(&Metronome<PeriodType>::worker, this, callback);
            }

            inline
            bool is_running() { return _is_running; }

            void spin() {
                if (_is_running) {
                    throw std::runtime_error{"Metronome is already operating in standalone mode."};
                }
                internal_spin();
            }

        private:
            PeriodType period;
            std::atomic<bool> _is_running{false};
            std::thread working_thread;

            bool first{true};
            std::chrono::high_resolution_clock::time_point next_call;

            void worker(std::function<void()> callback)
            {
                while (_is_running) {
                    callback();
                    internal_spin();
                }
            }

            void internal_spin()
            {
                if (period.count() == 0) {
                    return;
                }

                if (first) {
                    first = false;
                    next_call = std::chrono::high_resolution_clock::now() + period;
                    return;
                }

                std::this_thread::sleep_until(next_call);
                next_call = next_call + period;
            }
    };
}

#endif /* RL_CPPUTILS_METRONOME_H_ */
