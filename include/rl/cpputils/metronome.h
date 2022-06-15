#ifndef RL_CPPUTILS_METRONOME_H_
#define RL_CPPUTILS_METRONOME_H_


#include <chrono>
#include <thread>
#include <functional>
#include <atomic>


namespace rl::cpputils
{
    /**
     * @brief Time period keeping class.
     * 
     * This class can be interacted with using either the `start` method, that will
     * execute a given function repeatedly, or the `spin` method, that will block the
     * calling thread such that repeated calls are separated by the correct period of
     * time.
     * 
     * @tparam PeriodType Time unit.
     */
    template<typename PeriodType = std::chrono::seconds>
    class Metronome
    {
        public:
            /**
             * @brief Construct a new Metronome object.
             * 
             * @param period Time period.
             */
            Metronome(size_t period)
            : period{PeriodType(period)} {}

            ~Metronome() {
                stop();
            }

            /**
             * @brief Stops the metronome, if ever started.
             */
            void stop() {
                _is_running = false;
                if (working_thread.joinable()) {
                    working_thread.join();
                }
            }

            /**
             * @brief Starts the metronome, calling the given function repeatedly with
             * the given period.
             * 
             * @param callback Function to be executed.
             */
            void start(std::function<void()> callback) {
                if (_is_running) throw std::runtime_error{"Metronome already in a running state."};
                _is_running = true;
                working_thread = std::thread(&Metronome<PeriodType>::worker, this, callback);
            }

            /**
             * @return true if the metronome is running.
             * @return false if the metronome is not running.
             */
            inline bool is_running() { return _is_running; }

            /**
             * @brief Blocks until the specified period has passed since the last call.
             * The first call blocks for exactly period units of time.
             * 
             */
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
