import numpy as np
import time
from multiprocessing import shared_memory


def create_shared_array(input_array, shape, dtype, name="shared_points"):
    array_size = np.prod(shape) * np.dtype(dtype).itemsize
    try:
        # Create a shared array
        shm = shared_memory.SharedMemory(create=True, size=array_size, name=name)
    except FileExistsError:
        print(f"Shared memory with name {name} already exists.")
        # Clean up the shared memory if it already exists
        shm = shared_memory.SharedMemory(create=False, size=array_size, name=name)
    except Exception as e:
        print(f"Error creating shared memory: {e}")
        raise e

    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    # Fill array with input data
    arr.fill(0)  # Initialize the array with zeros
    arr[:] = input_array[:]
    return arr, shm

def attach_shared_array(shape, dtype, name="shared_points"):
    while True:
        try:
            # Attach to an existing shared array
            shm = shared_memory.SharedMemory(name=name, create=False)
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            assert arr.shape == shape, f"Expected shape {shape} but got {arr.shape}"
            assert arr.dtype == dtype, f"Expected dtype {dtype} but got {arr.dtype}"
            print("Attached to shared memory")
            print(f"Memory state: size={shm.size}, name={shm.name}")
            print(f"Array details: shape={arr.shape}, dtype={arr.dtype}, strides={arr.strides}, offset={arr.__array_interface__['data'][0] - shm.buf.tobytes().find(arr.tobytes())}")

            return arr, shm
        except FileNotFoundError:
            time.sleep(0.2)

def main():
    # sample array
    input_array = np.random.rand(10, 10).astype(np.float32)
    shape = input_array.shape
    dtype = input_array.dtype

    # Create shared array
    shared_array, shm = create_shared_array(input_array, shape, dtype)
    print(f"Shared array: {shared_array[0,1]}")

    # Attach to shared array
    attached_array, attached_shm = attach_shared_array(shape, dtype)
    print(f"Attached array: {attached_array[0,1]}")

    # Detach from shared memory
    attached_shm.close()
    shm.close()
    print("Detached from shared memory")

    # Clean up shared memory
    shm.unlink()
    print("Cleaned up shared memory")

main()