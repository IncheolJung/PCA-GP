from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def parallel_function(x):
    """
    Replace this with the function you want to parallelize.
    It should accept a single input (or a tuple of inputs).
    """
    # Example: square the input
    return x**2


def chunk_data(data, size):
    """Split data into roughly equal chunks for each process."""
    avg = len(data) // size
    chunks = []
    for i in range(size):
        start = i * avg
        end = (i + 1) * avg if i != size - 1 else len(data)
        chunks.append(data[start:end])
    return chunks


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Step 1: Only rank 0 prepares the data
    if rank == 0:
        data = list(range(1, 21))  # Example data
        chunks = chunk_data(data, size)
    else:
        chunks = None

    # Step 2: Scatter chunks to all processes
    local_chunk = comm.scatter(chunks, root=0)

    # Step 3: Each process computes its results
    # local_index  = [i for i in range(len(local_chunk))]
    # local_result = [parallel_function(x) for x in local_chunk]
    local_dict = {i: parallel_function(x) for i, x in enumerate(local_chunk)}
    local_index, local_result = zip(*local_dict.items())

    # Step 4: Gather results back to root
    results_idx = comm.gather(local_index, root=0)
    results = comm.gather(local_result, root=0)

    # Step 5: Only root combines and prints results
    if rank == 0:
        # Flatten the list
        idx = [item for sublist in results_idx for item in sublist]
        results = [item for sublist in results for item in sublist]
        results_dict = {i:d for i,d in zip(idx, results)}
        print("Final Results:", results)