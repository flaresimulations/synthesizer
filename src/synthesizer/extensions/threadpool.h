/*******************************************************************************
 * This file is part of MEGA++.
 * Copyright (c) 2023 Will Roper (w.roper@sussex.ac.uk)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * This header file contains the defintion of the threadpool class used to
 * distribute local work over local threads. This implementation uses pthreads.
 ******************************************************************************/
#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#if __cplusplus < 201703L
#error "This code requires C++17 or later."
#endif

// Keys for thread specific data.
extern pthread_key_t threadpool_tid;

/**
 * @brief The ThreadPool class provides a flexible and efficient mechanism
 * for parallelizing tasks across multiple threads using pthreads.
 *
 * The class is designed to distribute local workloads over local threads,
 * offering a simplified interface for parallelizing the execution of a
 * specified function over a given array of data. The implementation ensures
 * synchronization and coordination among the worker threads.
 *
 * Functions mapped over must have the following signature:
 *   void mapFunction(void *mapData, int size, void *extraData)
 *
 * Key Features:
 * - Automatic and uniform chunking options for workload distribution.
 * - Support for additional data to be passed to the map function.
 * - Dynamic control over the number of worker threads in the pool.
 * - Thread safety and efficient handling of synchronization using
 *   condition variables and atomic operations.
 *
 * Usage:
 * - Create an instance of the ThreadPool with the desired number of threads.
 * - Use the `map` function to apply a given function to an array of data
 *   in parallel, providing options for chunking and additional data.
 * - The class ensures proper initialization, cleanup, and coordination of
 *   worker threads through its constructor and destructor.
 */
class ThreadPool {
public:
  // Constants
  static const int threadpool_auto_chunk_size = 0;
  static const int threadpool_uniform_chunk_size = -1;
  static const int threadpool_default_chunk_ratio = 4;

private:
  // Number of threads
  int numThreads;

  // Define the index of the current threadpool task.
  std::atomic<size_t> taskInd;

  // Member variables
  std::vector<std::thread> threads;
  std::mutex waitMutex, runMutex;
  std::condition_variable waitCondition, runCondition;

  // Map function and data
  std::function<void(void *, int, void *)> mapFunction;
  void *mapData;
  size_t mapDataSize;
  size_t mapDataCount;
  size_t mapDataStride;
  size_t mapDataChunk;
  void *mapExtraData;

  // Number of threads running and finished
  std::atomic<int> numThreadsRunning;
  std::atomic<int> numThreadsFinished;

  // Flag for when we are done mapping
  bool done;

  std::mutex coutMutex;

public:
  // Constructor
  ThreadPool(int numThreads);

  // Destructor
  ~ThreadPool();

  // Map function to apply a given function to an array of data in parallel
  void map(std::function<void(void *, int, void *)> mapFunction, void *mapData,
           size_t dataSize, int chunk, void *extraData = nullptr);

private:
  // Struct to store log entry information
  struct LogEntry {
    int tid;
    size_t chunkSize;
    std::function<void(void *, int, void *)> mapFunction;
    std::chrono::steady_clock::time_point tic, toc;
  };

  // Struct to store log entries for each thread
  struct MapperLog {
    std::vector<LogEntry> log;
    int count;
  };

  // Helper function to initialize threads
  void initializeThreads();

  // Worker thread function
  void workerThread(int tid);
};

// Declare and define the threadpool_tid globally.
pthread_key_t threadpool_tid;

/**
 * @brief Constructor for the ThreadPool class.
 *
 * @param numThreads The number of worker threads in the thread pool.
 */
ThreadPool::ThreadPool(int numThreads)
    : taskInd(0), numThreads(numThreads - 1), numThreadsRunning(0) {
  initializeThreads();
}

/**
 * @brief Cleans up the resources and terminates the worker threads.
 *
 * This function sets the mapFunction to nullptr, signals that the threads
 * should be terminated, wakes up the threads to notify them of the shutdown,
 * and then joins the threads to wait for their completion.
 */
ThreadPool::~ThreadPool() {
  // Clean up the threadpool
  this->mapFunction = nullptr;

  // We're done
  this->done = true;

  // Wake up the threads to let them know we're done
  runCondition.notify_all();

  // Join the threads
  for (auto &thread : this->threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

/**
 * @brief Initializes the worker threads in the thread pool.
 */
void ThreadPool::initializeThreads() {
  // Attach the threads
  for (int i = 0; i < this->numThreads; ++i) {
    this->threads.emplace_back(&ThreadPool::workerThread, this, i);
  }

  printf("Threadpool: %d threads created.\n", this->numThreads);

  // Initialise the flag to say we are done
  this->done = false;

  // Signal to the threads that it's sleep time until further notice
  std::unique_lock<std::mutex> lock(runMutex);
  waitCondition.wait(lock,
                     [&]() { return numThreadsRunning == this->numThreads; });
}

/**
 * @brief Applies a given function to an array of data in parallel.
 *
 * This function sets up the member variables and then signals the worker
 * threads to start processing. The actual work is done in the workerThread.
 *
 * TODO: The main thread should be able to do some work too.
 *
 * @param mapFunction The function to apply to each element of the array.
 * @param mapData Pointer to the array of data.
 * @param dataSize The size of the array.
 * @param chunk The defintion to use to define the size of each processing
 *              chunk.
 * @param extraData Additional data to be passed to the map function.
 */
void ThreadPool::map(std::function<void(void *, int, void *)> mapFunction,
                     void *mapData, size_t dataSize, int chunk,
                     void *extraData) {
  /* Handle the serial case. */
  if (this->numThreads == 0) {
    mapFunction(mapData, dataSize, extraData);
    return;
  }

  std::unique_lock<std::mutex> lock(waitMutex);

  // Set the map function
  this->mapFunction = mapFunction;

  // Set the map data and size
  this->mapData = mapData;
  this->mapDataSize = dataSize;

  // Set the extra data
  this->mapExtraData = extraData;

  // Reset the threadpool counts and indices
  this->mapDataCount = 0;
  this->numThreadsFinished = 0;
  this->taskInd = 0;

  // Set the chunking approach
  if (chunk == threadpool_auto_chunk_size) {
    this->mapDataChunk = std::max<size_t>(
        dataSize / (this->numThreads * threadpool_default_chunk_ratio), 1U);
  } else if (chunk == threadpool_uniform_chunk_size) {
    this->mapDataChunk = threadpool_uniform_chunk_size;

  } else {
    this->mapDataChunk = chunk;
  }

  // Make sure the chunk size is valid
  if (this->mapDataChunk < 1) {
    this->mapDataChunk = 1;
  }
  if (this->mapDataChunk > INT_MAX) {
    this->mapDataChunk = INT_MAX;
  }

  // Let the threads know we are ready to run
  runCondition.notify_all();

  // Wait for all threads to finish their work
  waitCondition.wait(
      lock, [&]() { return this->numThreadsFinished == this->numThreads; });
}

/**
 * @brief Worker thread function for the ThreadPool class.
 *
 * This is the actual work function.
 *
 * Workers will go to sleep if there's no work to do and then wake up when
 * signalled by the main thread to do so.
 *
 * Workers
 *
 * @param tid The thread ID of the worker thread.
 */
void ThreadPool::workerThread(int tid) {

  // When we first start we need to signal we're alive and then sleep
  // until we're needed.
  {
    std::unique_lock<std::mutex> lock(runMutex);
    numThreadsRunning.fetch_add(1, std::memory_order_relaxed);
    waitCondition.notify_one();
    runCondition.wait(lock);
  }

  // Set the thread ID for this thread
  int localtid = tid;
  pthread_setspecific(threadpool_tid, &localtid);

  // Define the current threadpool task index.
  size_t currentTaskInd;

  // Keep on, keeping on... until the program exits
  while (true) {

    // Set the chunk size for this job
    ptrdiff_t chunkSize = this->mapDataChunk;

    // Get the current task index
    currentTaskInd = taskInd.fetch_add(chunkSize, std::memory_order_relaxed);

    // Have we finished all the current work?
    if (currentTaskInd >= this->mapDataSize) {

      // We have, signal that this thread is done
      std::unique_lock<std::mutex> waitLock(waitMutex);
      numThreadsRunning.fetch_sub(1, std::memory_order_relaxed);
      numThreadsFinished.fetch_add(1, std::memory_order_relaxed);

      // Signal that, we're waiting and then have a nap
      waitCondition.notify_one();
      runCondition.wait(waitLock);

      // We're awake
      numThreadsRunning.fetch_add(1, std::memory_order_relaxed);

      // Get the new task index and chunk size
      chunkSize = this->mapDataChunk;
      currentTaskInd = taskInd.fetch_add(chunkSize, std::memory_order_relaxed);
    }

    // Are we done? (triggered when the destructor is called)
    if (this->done) {
      break;
    }

    // Get the pointer to the current map data and current chunk size
    double *currentMapData =
        &static_cast<double *>(this->mapData)[currentTaskInd];

    // Handle chunks that would extend beyond the end of the map data
    ptrdiff_t currentChunkSize = std::min(
        static_cast<ptrdiff_t>(mapDataSize - currentTaskInd), chunkSize);

    printf("Thread %d: Processing chunk %d of size %d (of %d)\n", tid,
           currentTaskInd, currentChunkSize, mapDataSize);

    // Call the map function
    mapFunction(static_cast<void *>(currentMapData), currentChunkSize,
                this->mapExtraData);
  }
}

#endif // THREADPOOL_H
