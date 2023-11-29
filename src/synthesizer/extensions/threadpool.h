/******************************************************************************
 * A C module containing the definition and functions used for running a
 * pthread based threadpool.
 *****************************************************************************/

/* C includes */
#include <pthread.h>
#include <stdlib.h>

/* Function signature for the map function */
typedef void (*MapFunction)(void *map_data, int n_elements, void *extra_data);

/* Structure to hold task information */
typedef struct {
  void *map_data;
  int n_elements;
  void *extra_data;
} Task;

/* Structure to represent the thread pool */
typedef struct {
  pthread_t *threads;
  int nthreads;
  Task *task_queue;
  int queue_size;
  int front;
  int rear;
  pthread_mutex_t mutex;
  pthread_cond_t condition;
  int shutdown;
} ThreadPool;

/* Forward declaration of workerThread */
void *workerThread(ThreadPool *pool);

/* Initialize the thread pool */
ThreadPool *initializeThreadPool(int nthreads, int queue_size) {
  ThreadPool *pool = (ThreadPool *)malloc(sizeof(ThreadPool));
  pool->threads = (pthread_t *)malloc(nthreads * sizeof(pthread_t));
  pool->nthreads = nthreads;
  pool->task_queue = (Task *)malloc(queue_size * sizeof(Task));
  pool->queue_size = queue_size;
  pool->front = -1;
  pool->rear = -1;
  pool->shutdown = 0;

  pthread_mutex_init(&pool->mutex, NULL);
  pthread_cond_init(&pool->condition, NULL);

  /* Create worker threads */
  for (int i = 0; i < nthreads; ++i) {
    pthread_create(&pool->threads[i], NULL, (void *(*)(void *))workerThread,
                   pool);
  }

  return pool;
}

/* Worker thread function */
void *workerThread(ThreadPool *pool) {
  while (1) {
    pthread_mutex_lock(&pool->mutex);

    /* Wait for a task to be added to the queue */
    while (pool->front == -1 && pool->shutdown == 0) {
      pthread_cond_wait(&pool->condition, &pool->mutex);
    }

    /* Check if the thread should exit */
    if (pool->shutdown) {
      pthread_mutex_unlock(&pool->mutex);
      pthread_exit(NULL);
    }

    /* Dequeue a task from the queue */
    Task task = pool->task_queue[pool->front];
    if (pool->front == pool->rear) {
      pool->front = -1;
      pool->rear = -1;
    } else {
      pool->front = (pool->front + 1) % pool->queue_size;
    }

    pthread_mutex_unlock(&pool->mutex);

    /* Execute the map function with the task data */
    MapFunction mapFunction = (MapFunction)task.map_data;
    mapFunction(task.map_data, task.n_elements, task.extra_data);
  }

  pthread_exit(NULL);
}

/* Add a task to the thread pool */
void submitTask(ThreadPool *pool, void *map_data, int n_elements,
                void *extra_data) {
  pthread_mutex_lock(&pool->mutex);

  /* Check if the queue is full */
  while ((pool->rear + 1) % pool->queue_size == pool->front) {
    pthread_cond_wait(&pool->condition, &pool->mutex);
  }

  /* Enqueue the task */
  if (pool->front == -1 && pool->rear == -1) {
    pool->front = 0;
    pool->rear = 0;
  } else {
    pool->rear = (pool->rear + 1) % pool->queue_size;
  }

  pool->task_queue[pool->rear].map_data = map_data;
  pool->task_queue[pool->rear].n_elements = n_elements;
  pool->task_queue[pool->rear].extra_data = extra_data;

  /* Signal a waiting thread that a task is available */
  pthread_cond_signal(&pool->condition);

  pthread_mutex_unlock(&pool->mutex);
}

/* Shut down the thread pool */
void shutdownThreadPool(ThreadPool *pool) {
  pthread_mutex_lock(&pool->mutex);

  /* Set shutdown flag */
  pool->shutdown = 1;

  /* Signal all waiting threads to exit */
  pthread_cond_broadcast(&pool->condition);

  pthread_mutex_unlock(&pool->mutex);

  /* Wait for all threads to finish */
  for (int i = 0; i < pool->nthreads; ++i) {
    pthread_join(pool->threads[i], NULL);
  }

  /* Clean up resources */
  free(pool->threads);
  free(pool->task_queue);
  free(pool);

  pthread_mutex_destroy(&pool->mutex);
  pthread_cond_destroy(&pool->condition);
}

/* Threadpool mapper function */
void threadpoolMapper(int nthreads, void *map_data, int n_elements,
                      void *extra_data, MapFunction mapFunction) {
  ThreadPool *pool =
      initializeThreadPool(nthreads, 10); /* You can adjust the queue size */

  /* Break down the mapping into tasks and submit them to the thread pool */
  for (int i = 0; i < n_elements; ++i) {
    submitTask(pool, map_data, n_elements, extra_data);
  }

  /* Wait for all tasks to complete */
  shutdownThreadPool(pool);
}
