//! # oxide-pipeline
//!
//! Full 5-layer pipeline simulation:
//! Intent → Pincher → Flux → cuda-oxide → cudaclaw
//! with conservation verification at each stage.

use std::collections::VecDeque;

// ─── Layer 1: Intent (open-parallel async) ───

#[derive(Debug, Clone)]
pub struct Intent {
    pub description: String,
    pub input_data: Vec<i8>,
}

pub fn submit_intent(desc: &str, data: Vec<i8>) -> Intent {
    Intent { description: desc.into(), input_data: data }
}

// ─── Layer 2: Pincher (intent → ops) ───

#[derive(Debug, Clone, PartialEq)]
pub enum FluxOp {
    TAdd { rd: usize, ra: usize, rb: usize },
    TMul { rd: usize, ra: usize, rb: usize },
    TNeg { rd: usize, ra: usize },
    Sync,
    Halt,
}

pub fn compile_intent(intent: &Intent) -> Vec<FluxOp> {
    if intent.description.contains("reduce") {
        vec![
            FluxOp::TAdd { rd: 0, ra: 0, rb: 1 },
            FluxOp::Sync,
            FluxOp::Halt,
        ]
    } else if intent.description.contains("filter") {
        vec![
            FluxOp::TMul { rd: 0, ra: 0, rb: 2 }, // mask
            FluxOp::Halt,
        ]
    } else {
        vec![
            FluxOp::TNeg { rd: 0, ra: 0 },
            FluxOp::Sync,
            FluxOp::Halt,
        ]
    }
}

// ─── Layer 3: Flux VM (bytecode execution) ───

pub struct FluxVM {
    pub registers: Vec<i8>,
    pub trace: Vec<String>,
}

impl FluxVM {
    pub fn new(data: &[i8]) -> Self {
        let mut registers = vec![0i8; 16];
        for (i, &v) in data.iter().take(16).enumerate() { registers[i] = v; }
        Self { registers, trace: Vec::new() }
    }

    pub fn execute(&mut self, ops: &[FluxOp]) -> Vec<i8> {
        for op in ops {
            match op {
                FluxOp::TAdd { rd, ra, rb } => {
                    let a = self.registers.get(*ra).copied().unwrap_or(0);
                    let b = self.registers.get(*rb).copied().unwrap_or(0);
                    self.registers[*rd] = tadd(a, b);
                    self.trace.push(format!("TADD r{}, r{}, r{} → {}", rd, ra, rb, self.registers[*rd]));
                }
                FluxOp::TMul { rd, ra, rb } => {
                    let a = self.registers.get(*ra).copied().unwrap_or(0);
                    let b = self.registers.get(*rb).copied().unwrap_or(0);
                    self.registers[*rd] = tmul(a, b);
                    self.trace.push(format!("TMUL r{}, r{}, r{} → {}", rd, ra, rb, self.registers[*rd]));
                }
                FluxOp::TNeg { rd, ra } => {
                    let a = self.registers.get(*ra).copied().unwrap_or(0);
                    self.registers[*rd] = -a;
                    self.trace.push(format!("TNEG r{}, r{} → {}", rd, ra, self.registers[*rd]));
                }
                FluxOp::Sync => { self.trace.push("SYNC".into()); }
                FluxOp::Halt => { self.trace.push("HALT".into()); break; }
            }
        }
        self.registers.clone()
    }
}

// ─── Layer 4: cuda-oxide (conservation verification) ───

pub fn verify_conservation(input: &[i8], output: &[i8]) -> bool {
    let in_sum: i32 = input.iter().map(|&v| v as i32).sum();
    let out_sum: i32 = output.iter().map(|&v| v as i32).sum();
    (in_sum - out_sum).abs() <= input.len() as i32
}

// ─── Layer 5: cudaclaw (GPU dispatch simulation) ───

#[derive(Debug, Clone)]
pub struct GpuDispatch {
    pub kernel_name: String,
    pub threads: u32,
    pub output: Vec<i8>,
    pub execution_us: u64,
}

pub fn dispatch_gpu(name: &str, data: &[i8]) -> GpuDispatch {
    GpuDispatch {
        kernel_name: name.into(),
        threads: data.len() as u32 * 32,
        output: data.to_vec(),
        execution_us: data.len() as u64 * 4,
    }
}

// ─── Full Pipeline ───

#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub intent: String,
    pub ops: Vec<FluxOp>,
    pub vm_trace: Vec<String>,
    pub conserved: bool,
    pub gpu_output: Vec<i8>,
    pub total_time_us: u64,
    pub layers_used: usize,
}

pub fn run_pipeline(intent: &Intent) -> PipelineResult {
    // Layer 1: Intent already submitted
    // Layer 2: Pincher compiles
    let ops = compile_intent(intent);
    // Layer 3: Flux VM executes
    let mut vm = FluxVM::new(&intent.input_data);
    let output = vm.execute(&ops);
    // Layer 4: Conservation check
    let conserved = verify_conservation(&intent.input_data, &output);
    // Layer 5: GPU dispatch
    let gpu = dispatch_gpu("ternary_kernel", &output);

    PipelineResult {
        intent: intent.description.clone(),
        ops, vm_trace: vm.trace, conserved,
        gpu_output: gpu.output,
        total_time_us: gpu.execution_us,
        layers_used: 5,
    }
}

/// Run batch pipeline with multiple intents.
pub fn run_batch(intents: &[Intent]) -> Vec<PipelineResult> {
    intents.iter().map(run_pipeline).collect()
}

// ─── Z₃ arithmetic ───

fn tadd(a: i8, b: i8) -> i8 {
    match (a, b) {
        (-1, -1) => 1, (-1, 0) => -1, (-1, 1) => 0,
        (0, -1) => -1, (0, 0) => 0, (0, 1) => 1,
        (1, -1) => 0, (1, 0) => 1, (1, 1) => -1, _ => 0,
    }
}

fn tmul(a: i8, b: i8) -> i8 {
    match (a, b) {
        (-1, -1) => 1, (-1, 1) => -1, (1, -1) => -1, (1, 1) => 1, _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_pipeline_reduce() {
        let intent = submit_intent("reduce the data", vec![1, -1, 1, -1]);
        let result = run_pipeline(&intent);
        assert!(result.conserved);
        assert!(result.total_time_us > 0);
        assert_eq!(result.layers_used, 5);
    }

    #[test]
    fn test_full_pipeline_filter() {
        let intent = submit_intent("filter with mask", vec![1, 0, -1, 1]);
        let result = run_pipeline(&intent);
        assert_eq!(result.layers_used, 5);
    }

    #[test]
    fn test_full_pipeline_negate() {
        let intent = submit_intent("transform the data", vec![1, -1, 0]);
        let result = run_pipeline(&intent);
        assert_eq!(result.gpu_output[0], -1); // negated
    }

    #[test]
    fn test_vm_execution() {
        let mut vm = FluxVM::new(&[1, -1, 0, 1]);
        let ops = vec![
            FluxOp::TAdd { rd: 0, ra: 0, rb: 1 },
            FluxOp::Halt,
        ];
        let output = vm.execute(&ops);
        assert_eq!(output[0], 0); // tadd(1, -1) = 0
    }

    #[test]
    fn test_conservation_check() {
        assert!(verify_conservation(&[1, -1], &[0, 0]));
        assert!(verify_conservation(&[1, 1], &[1, 1]));
    }

    #[test]
    fn test_batch_pipeline() {
        let intents = vec![
            submit_intent("reduce", vec![1, -1]),
            submit_intent("filter", vec![1, 0]),
            submit_intent("transform", vec![-1, 1]),
        ];
        let results = run_batch(&intents);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_dispatch_simulation() {
        let gpu = dispatch_gpu("test", &[1, -1, 0, 1]);
        assert_eq!(gpu.threads, 128);
        assert!(gpu.execution_us > 0);
    }
}
