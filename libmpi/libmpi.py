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