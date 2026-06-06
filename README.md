# oxide-pipeline

Experiment: full 5-layer pipeline simulation. Intent→pincher→flux→cuda-oxide→cudaclaw end-to-end with conservation verification at each stage.

## Why This Matters

# oxide-pipeline
Full 5-layer pipeline simulation:
Intent → Pincher → Flux → cuda-oxide → cudaclaw
with conservation verification at each stage.

## The Five-Layer Stack

This crate is part of the **Oxide Stack** — a distributed GPU runtime built on five layers:

```
┌─────────────────┐
│  cudaclaw        │  Persistent GPU kernels, warp consensus, SmartCRDT
├─────────────────┤
│  cuda-oxide      │  Flux → MIR → Pliron → NVVM → PTX compiler
├─────────────────┤
│  flux-core       │  Bytecode VM + A2A agent protocol
├─────────────────┤
│  pincher         │  "Vector DB as runtime, LLM as compiler"
├─────────────────┤
│  open-parallel   │  Async runtime (tokio fork)
└─────────────────┘
```

The key insight: **ternary values {-1, 0, +1} map directly to GPU compute**. They pack 16× denser than FP32, enable XNOR+popcount matmul, and conservation laws become compile-time checks.

## Design

Every value in this crate follows **ternary algebra** (Z₃):

| Value | Meaning | GPU Analog |
|-------|---------|------------|
| +1 | Positive / Active / Healthy | Warp vote yes |
| 0 | Neutral / Pending / Balanced | Warp vote abstain |
| -1 | Negative / Failed / Overloaded | Warp vote no |

This isn't arbitrary — ternary is the natural encoding for:
1. **BitNet b1.58** (Microsoft) — ternary LLMs at 60% less power
2. **GPU warp voting** — hardware ballot returns ternary consensus
3. **Conservation laws** — {-1, 0, +1} preserves quantity

## Key Types

```rust
pub struct Intent
pub fn submit_intent
pub enum FluxOp
pub fn compile_intent
pub struct FluxVM
pub fn new
pub fn execute
pub fn verify_conservation
pub struct GpuDispatch
pub fn dispatch_gpu
pub struct PipelineResult
pub fn run_pipeline
```

## Usage

```toml
[dependencies]
oxide-pipeline = "0.1.0"
```

```rust
use oxide_pipeline::*;
// See src/lib.rs tests for complete working examples
```

## Testing

```bash
git clone https://github.com/SuperInstance/oxide-pipeline.git
cd oxide-pipeline
cargo test    # 7 tests
```

## Stats

| Metric | Value |
|--------|-------|
| Tests | 7 |
| Lines of Rust | 239 |
| Public API | 13 items |

## License

Apache-2.0
