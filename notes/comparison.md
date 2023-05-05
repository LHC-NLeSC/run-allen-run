# Compare event rate in Allen

![event rate vs batch sizes](./evt-rate-vs-batch-size-comparison.png "Event rate vs batch size")

The above plot shows the event rate against the batch size for the
following scenarios:
- multiple instances of `ghostbuster` (1-5)
- `baseline` is a `ghostbuster` run with inference skipped
- load models of 4 different sizes: `ghost_nn{,big,bigger,tiny}`
  - shown on different rows
- runs with the `ghost_nn` model also compare a `handcoded` version of
  the algorithm.  It includes a few simple optimisations.
- TensorRT runs where FP16 optimisations have been enabled are shown
  on the 2nd column
