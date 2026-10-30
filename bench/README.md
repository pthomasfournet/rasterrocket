# bench/ — historical benchmark results

Everything in this directory is a **historical record**. Each subdirectory
holds the raw output and summary of one benchmark run taken while a past
release was being prepared (v0.6.0 to v0.9.1, all in May 2026). None of it
describes the current renderer, and none of it has been re-run since.

Current benchmark results, and how to reproduce them, are on the project
wiki's [Benchmarks](../wiki/Benchmarks.md) page.

## What is here

| Directory | Run |
|---|---|
| `v060/` | v0.6.0 GPU baseline, full corpus; `results-testbench-i7-8700K-rtx2080super.md` is the same matrix on the reference testbench |
| `v070/`, `v070-testbench/` | v0.7.0 image-cache matrix, on the dev box and on the reference testbench |
| `v07x-baselines/` | pdftoppm and mutool timings on the corpus, taken for the v0.7.x comparisons |
| `v080/` | v0.8.0 gate for the parallel-Huffman GPU JPEG path |
| `v090/`, `v091/` | CPU vs CUDA vs Vulkan, standard and parallel-Huffman JPEG decode |
| `v10/` | CUDA vs Vulkan backend gate on the vector-heavy corpora 01–05 |
| `v11*/` | Million-page-archive contest (first pixel, sustained, cross-document, random access), CPU and GPU variants |

## Reading these results

- **Hardware.** "Dev box" runs used an AMD Ryzen 9 9900X3D with an RTX 5070.
  The "testbench" runs used an Intel i7-8700K with an RTX 2080 SUPER; that
  machine was retired in July 2026 and the testbench numbers cannot be
  reproduced.
- **Software.** The binaries measured were the release of the directory's
  name, or the commit named in the file (`v10` and `v11*` are named after
  the roadmap stage they gated, not a release). Rendering, decoding and GPU dispatch
  have changed substantially since; in particular nvJPEG, the parallel-Huffman
  GPU JPEG path and the GPU AA fill are all dormant by default today.
- **Scripts.** The files name the scripts that produced them
  (`scripts/bench_v0*.sh`, `tests/bench_corpus.sh`, `scripts/run_contest_v11*.sh`
  and their aggregators). Those scripts targeted the retired package name and
  have been removed from the tree; they remain in git history.
