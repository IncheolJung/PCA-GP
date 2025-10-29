from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

TAG_WORK = 1
TAG_DONE = 2
TAG_STOP = 3
TAG_REQUEST = 4

def do_work(task_id):
    print(f"[Rank {rank}] Working on task {task_id}")
    # simulate long task
    time.sleep(1 + rank * 0.1)
    return task_id * 2  # example result

if __name__ == "__main__":

    if rank == 0:
        # --- MASTER ---
        tasks = list(range(20))  # e.g. 20 long tasks
        ntasks = len(tasks)
        num_workers = size - 1
        finished_workers = 0

        # Give one task to each worker and keep one for self
        for dest in range(1, size):
            if tasks:
                task = tasks.pop(0)
                comm.send(task, dest=dest, tag=TAG_WORK)
            else:
                comm.send(None, dest=dest, tag=TAG_STOP)
                finished_workers += 1

        # Master starts working too
        if tasks:
            my_task = tasks.pop(0)
            result = do_work(my_task)
            print(f"[Master] Finished task {my_task} result={result}")
        else:
            my_task = None

        # Process results and assign new work dynamically
        while finished_workers < num_workers:
            status = MPI.Status()
            msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            src = status.Get_source()
            tag = status.Get_tag()

            if tag == TAG_DONE:
                result, task_id = msg
                print(f"[Master] Got result {result} from rank {src}")

                # Assign next task if available
                if tasks:
                    new_task = tasks.pop(0)
                    comm.send(new_task, dest=src, tag=TAG_WORK)
                else:
                    comm.send(None, dest=src, tag=TAG_STOP)
                    finished_workers += 1

        print("[Master] All work completed.")

    else:
        # --- WORKERS ---
        while True:
            task = comm.recv(source=0, tag=MPI.ANY_TAG)
            tag = MPI.Status().Get_tag()  # Not strictly needed here
            if task is None:
                break  # stop
            result = do_work(task)
            comm.send((result, task), dest=0, tag=TAG_DONE)

        print(f"[Rank {rank}] Terminating.")
