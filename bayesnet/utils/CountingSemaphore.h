#ifndef COUNTING_SEMAPHORE_H
#define COUNTING_SEMAPHORE_H
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <thread>

class CountingSemaphore {
public:
    static CountingSemaphore& getInstance()
    {
        static CountingSemaphore instance;
        return instance;
    }
    // Delete copy constructor and assignment operator
    CountingSemaphore(const CountingSemaphore&) = delete;
    CountingSemaphore& operator=(const CountingSemaphore&) = delete;
    void acquire()
    {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this]() { return count_ > 0; });
        --count_;
    }
    void release()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        ++count_;
        if (count_ <= max_count_) {
            cv_.notify_one();
        }
    }
    uint getCount() const
    {
        return count_;
    }
    uint getMaxCount() const
    {
        return max_count_;
    }
private:
    CountingSemaphore()
        : max_count_(std::max(1u, static_cast<uint>(0.95 * std::thread::hardware_concurrency()))),
        count_(max_count_)
    {
    }
    std::mutex mtx_;
    std::condition_variable cv_;
    const uint max_count_;
    uint count_;
};
#endif