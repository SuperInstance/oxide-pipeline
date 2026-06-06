# oxide-pipeline

Full five-layer GPU execution pipeline: Intent → Pincher → Flux → cuda-oxide → cudaclaw.

## Why This Exists

Most GPU programming models make you think about threads, blocks, and shared memory from line one. That's the wrong abstraction for most workloads. What you actually want is: "reduce this tensor," "filter these values," "transform this data." The gap between intent and hardware is five layers deep, and each layer has a specific job.

This crate implements all five as a single composable pipeline. Each layer has a clear contract: intent captures what you want, Pincher compiles it to operations, Flux executes on a ternary VM, cuda-oxide verifies conservation laws, and cudaclaw dispatches to the GPU. You can test each layer independently, or run the whole thing end-to-end.

## Architecture

```
Layer 1: Intent (open-parallel async)
  "reduce the data" ──────────────────────────────────────────────┐
  input_data: Vec<i8>                                              │
                                                                   ▼
Layer 2: Pincher (compiler)                              ┌─────────────────┐
  compile_intent() ──→ Vec<FluxOp>                       │ Intent → Ops    │
  "reduce" → [TAdd, Sync, Halt]                          │ Pattern-match   │
  "filter" → [TMul, Halt]                                │ on description  │
  default  → [TNeg, Sync, Halt]                          └────────┬────────┘
                                                                   ▼
Layer 3: Flux VM (bytecode execution)                    ┌─────────────────┐
  16 registers, Z₃ arithmetic                           │ Execute ops     │
  TAdd(a,b): (a+b) mod 3 mapped to {-1,0,+1}            │ on ternary data │
  TMul(a,b): (a×b) mapped to {-1,0,+1}                  │ 16 i8 registers │
  TNeg(a):   -a                                          └────────┬────────┘
  Sync / Halt                                                      ▼
Layer 4: cuda-oxide (conservation verification)          ┌─────────────────┐
  verify_conservation(input, output) → bool              │ Sum of input    │
  |input_sum - output_sum| ≤ len(input)                  │ ≈ sum of output │
                                                         └────────┬────────┘
                                                                   ▼
Layer 5: cudaclaw (GPU dispatch simulation)              ┌─────────────────┐
  dispatch_gpu(name, data) → GpuDispatch                 │ threads = n×32  │
  Simulated execution: data.len() × 4 μs                 │ output = data   │
                                                         └─────────────────┘
```

**Key types:**

- `Intent` — description + input data (Layer 1)
- `FluxOp` — ternary operations: `TAdd`, `TMul`, `TNeg`, `Sync`, `Halt` (Layer 2/3)
- `FluxVM` — 16-register ternary virtual machine (Layer 3)
- `GpuDispatch` — simulated GPU dispatch result (Layer 5)
- `PipelineResult` — complete pipeline output: trace, conservation status, GPU output

## Usage

```rust
use oxide_pipeline::*;

// Single pipeline run
let intent = submit_intent("reduce the data", vec![1, -1, 1, -1]);
let result = run_pipeline(&intent);

assert!(result.conserved);              // Conservation law verified
assert!(result.total_time_us > 0);      // GPU dispatch simulated
assert_eq!(result.layers_used, 5);      // All layers executed

println!("Trace: {:?}", result.vm_trace);
// ["TADD r0, r0, r1 → 0", "SYNC", "HALT"]

// Batch processing
let intents = vec![
    submit_intent("reduce with sum", vec![1, -1, 0, 1]),
    submit_intent("filter with mask", vec![1, 0, -1, 1]),
    submit_intent("transform data", vec![-1, 1, 0]),
];
let results = run_batch(&intents);
assert_eq!(results.len(), 3);

// Layer-by-layer control
let intent = submit_intent("test", vec![1, -1, 0, 1]);
let ops = compile_intent(&intent);               // Layer 2
let mut vm = FluxVM::new(&intent.input_data);    // Layer 3
let output = vm.execute(&ops);                   // Layer 3 execution
let conserved = verify_conservation(&intent.input_data, &output); // Layer 4
let gpu = dispatch_gpu("ternary_kernel", &output);               // Layer 5
```

## API Reference

### Layer 1: Intent

- `submit_intent(desc: &str, data: Vec<i8>) -> Intent`
- `struct Intent { description: String, input_data: Vec<i8> }`

### Layer 2: Pincher (Compiler)

- `compile_intent(intent: &Intent) -> Vec<FluxOp>` — pattern-matches description to operation sequence
- `enum FluxOp { TAdd{rd, ra, rb}, TMul{rd, ra, rb}, TNeg{rd, ra}, Sync, Halt }`

### Layer 3: Flux VM

- `FluxVM::new(data: &[i8]) -> Self` — initialize 16 registers with input data
- `execute(&mut self, ops: &[FluxOp]) -> Vec<i8>` — run operations, return final register state
- `registers: Vec<i8>` / `trace: Vec<String>`

### Layer 4: Conservation Verification

- `verify_conservation(input: &[i8], output: &[i8]) -> bool` — |Σinput - Σoutput| ≤ len(input)

### Layer 5: GPU Dispatch

- `dispatch_gpu(name: &str, data: &[i8]) -> GpuDispatch`
- `struct GpuDispatch { kernel_name, threads, output, execution_us }`

### Full Pipeline

- `run_pipeline(intent: &Intent) -> PipelineResult`
- `run_batch(intents: &[Intent]) -> Vec<PipelineResult>`
- `struct PipelineResult { intent, ops, vm_trace, conserved, gpu_output, total_time_us, layers_used }`

## The Deeper Idea

This is the **execution spine** of the oxide stack. The five layers map directly to the architecture's separation of concerns: *what* you want (Intent), *how* to express it (Pincher), *what happens* (Flux VM), *whether it's correct* (cuda-oxide verification), and *where it runs* (cudaclaw dispatch).

The Z₃ arithmetic (ternary values {-1, 0, +1} with modular addition) isn't arbitrary — it's the same algebra that underpins the energy conservation laws in oxide-energy-balance and the isolation quality signals in oxide-tenancy. Every operation in this pipeline is energy-conserving by construction, and the conservation check at Layer 4 verifies this at runtime.

## Related Crates

- **oxide-energy-balance** — formal verification that ternary operations preserve algebraic invariants
- **oxide-journal** — write-ahead log for pipeline state mutations
- **oxide-gradient** — optimizes kernel parameters that this pipeline executes
- **oxide-compile-cache** — caches compiled Flux bytecode to skip recompilation
