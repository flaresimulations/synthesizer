/******************************************************************************
 * A C module containing the definition and functions used for running a
 * pthread based threadpool.
 *****************************************************************************/

/* C includes */
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

/* Function signature for the map function */
typedef void (*MapFunction)(int start, int n_elements, void *extra_data);

/* Structure to hold task information */
typedef struct {
  int start;
  int n_elements;
  void *extra_data;
} Task;

struct ThreadData;

/* Structure to represent the thread pool */
typedef struct {
  pthread_t *threads;
  int nthreads;
  Task *task_queue;
  struct ThreadData *thread_data;
  int n_elements;
  MapFunction map_function;
  pthread_mutex_t barrier;
  pthread_cond_t condition;
  int shutdown;
  int completed_tasks;
} ThreadPool;

/* Structure to hold thread-specific data */
struct ThreadData {
  ThreadPool *pool;
  int thread_id;
};

void *workerThread(void *args) {

  /* Unpack our data */
  struct ThreadData *thread_data = (struct ThreadData *)args;
  ThreadPool *pool = thread_data->pool;
  int thread_id = thread_data->thread_id;

  /* Dequeue a task from the queue */
  Task *task = &pool->task_queue[thread_id];

  printf("%d: task->start=%d, task->n_elements=%d\n", thread_id, task->start,
         task->n_elements);

  /* Execute the map function with the task data */
  pool->map_function(task->start, task->n_elements, task->extra_data);

  pthread_exit(NULL);
}

/* Initialize the thread pool */
ThreadPool *initializeThreadPool(int nthreads, int n_elements,
                                 MapFunction map_function, void *extra_data) {

  ThreadPool *pool = (ThreadPool *)malloc(sizeof(ThreadPool));
  pool->threads = (pthread_t *)malloc(nthreads * sizeof(pthread_t));
  pool->nthreads = nthreads;
  pool->task_queue = (Task *)malloc(nthreads * sizeof(Task));
  pool->n_elements = n_elements;
  pool->shutdown = 0;
  pool->map_function = map_function;
  pool->thread_data =
      (struct ThreadData *)malloc(nthreads * sizeof(struct ThreadData));
  pool->completed_tasks = 0;

  /* Work out the chunk size to split the work equally. */
  int chunk_size = (int)(n_elements / nthreads);

  /* Create tasks for all threads. */
  for (int i = 0; i < nthreads; i++) {

    /* Enqueue the task */
    pool->task_queue[i].start = i * chunk_size;
    pool->task_queue[i].n_elements = chunk_size;
    pool->task_queue[i].extra_data = extra_data;
  }

  /* Add any left over elements onto the final thread.
   * NOTE: This could be done way better but our threadpool won't ever be
   *       complex enough that it should matter */
  pool->task_queue[nthreads - 1].n_elements +=
      (n_elements - (nthreads * chunk_size));

  int sum = 0;
  for (int i = 0; i < nthreads; i++)
    sum += pool->task_queue[i].n_elements;
  printf("Task element count=%d\n", sum);

  /* Initialise locks and conditions. */
  pthread_mutex_init(&pool->barrier, NULL);
  pthread_cond_init(&pool->condition, NULL);

  /* Create worker threads */
  for (int i = 0; i < nthreads; ++i) {
    pool->thread_data[i].pool = pool;
    pool->thread_data[i].thread_id = i;
    pthread_create(&pool->threads[i], NULL, (void *(*)(void *))workerThread,
                   &pool->thread_data[i]);
  }

  return pool;
}

/* Shut down the thread pool */
void shutdownThreadPool(ThreadPool *pool) {

  /* Wait for all threads to finish */
  for (int i = 0; i < pool->nthreads; ++i) {
    pthread_join(pool->threads[i], NULL);
  }

  /* Clean up resources */
  free(pool->threads);
  free(pool->task_queue);
  free(pool->thread_data);
  free(pool);

  pthread_mutex_destroy(&pool->barrier);
  pthread_cond_destroy(&pool->condition);
}

/* Threadpool mapper function */
void threadpoolMapper(int nthreads, int n_elements, void *extra_data,
                      MapFunction map_function) {

  /* If we are single threaded just call the map function directly. */
  if (nthreads == 1) {
    map_function(/*start*/ 0, n_elements, extra_data);
    return;
  }

  /* Intialise the threadpool itself.
   * This just sets up the threadpool struct, and populates it with POSIX
   * threads and the function being mapped over. */
  ThreadPool *pool =
      initializeThreadPool(nthreads, n_elements, map_function, extra_data);

  /* Wait for all tasks to complete */
  shutdownThreadPool(pool);
}
