# oxide-pipeline

Experiment: full 5-layer pipeline simulation. Intent→pincher→flux→cuda-oxide→cudaclaw end-to-end with conservation verification at each stage.

## Overview

# oxide-pipeline

Full 5-layer pipeline simulation:

## Stats

- **Tests**: 7
- **LOC**: 238
- **License**: Apache-2.0

## Part of the Oxide Stack

This crate is part of the [Flux→PTX](https://github.com/SuperInstance/cuda-oxide/blob/main/FLUX_TO_PTX.md) experimental suite, testing synergies between the five layers of the distributed GPU runtime:

1. **open-parallel** — async runtime (tokio fork)
2. **pincher** — "Vector DB as runtime, LLM as compiler"
3. **flux-core** — bytecode VM + A2A agent protocol
4. **cuda-oxide** — Flux→MIR→Pliron→NVVM→PTX compiler
5. **cudaclaw** — persistent GPU kernels, warp-level consensus, SmartCRDT

## Usage

```rust
use oxide_pipeline::*;
// See tests in src/lib.rs for examples
```

## License

Apache-2.0
