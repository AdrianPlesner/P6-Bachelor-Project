import multiprocessing as mp


def test(t):
    print("test")
    return t


if __name__ == '__main__':
    print("begin")
    n_proc = mp.cpu_count()
    with mp.Pool(processes=n_proc) as pool:
        proc_results = [pool.apply_async(test,
                                         args=(2,))
                        for _ in range(n_proc)]

        # blocks until all results are fetched
        result_chunks = [r.get() for r in proc_results]
    print("done")
