# Compare event rate in Allen

![event rate vs batch sizes (FP16)](./evt-rate-vs-batch-size-comparison-fp16.png "Event rate vs batch size (FP16)")

The above plot shows the event rate against the batch size for the
following scenarios:
- multiple instances of `ghostbuster` (1-5)
- `baseline` is a `ghostbuster` run with inference skipped
- load models of 4 different sizes: `ghost_nn{,_big,_bigger,_tiny}`
  - shown on different rows
- runs with the `ghost_nn` model also compare a `handcoded` version of
  the algorithm.  It includes a few simple optimisations.
- TensorRT runs where FP16 optimisations have been enabled are shown
  on the 2nd column

![event rate vs batch sizes (INT8)](./evt-rate-vs-batch-size-comparison-int8.png "Event rate vs batch size (INT8)")
