from mpi4py import MPI

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

def main():
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
    local_result = [parallel_function(x) for x in local_chunk]

    # Step 4: Gather results back to root
    results = comm.gather(local_result, root=0)

    # Step 5: Only root combines and prints results
    if rank == 0:
        # Flatten the list
        results = [item for sublist in results for item in sublist]
        print("Final Results:", results)

if __name__ == "__main__":
    main()
