# Matrix Library Usage Examples

The new `stratrix` crate provides a unified, generic matrix library with multiple multiplication strategies.

## Basic Usage

```rust
use stratrix::{Matrix, Strategy, MatrixConfig};
use rand::Rng;

fn main() {
    // Create random matrices with default config (Tiled strategy, block size 32)
    let limit = (6.0_f64 / 100.0).sqrt();
    let matrix_a = Matrix::random(100, 100, || {
        rand::thread_rng().gen_range(-limit..limit)
    });
    
    let matrix_b = Matrix::random(100, 100, || {
        rand::thread_rng().gen_range(-limit..limit)
    });
    
    // Create matrix with custom strategy and block size
    let config = MatrixConfig::new(Strategy::TiledParallel, 64);
    let matrix_c = Matrix::random_with_config(100, 100, || rand::random::<f64>(), config);
    
    // Perform operations - matrices use their configured strategy
    let add_result = matrix_a.add(&matrix_b).unwrap();
    let sub_result = matrix_b.subtract(&matrix_a).unwrap();
    let mul_result = matrix_a.multiply(&matrix_b).unwrap();
    let hadamard = matrix_a.hadamard_product(&matrix_b).unwrap();
    
    println!("Results work with the configured strategy!");
}
```

## Generic Over Numeric Types

```rust
use stratrix::Matrix;

fn main() {
    // Works with f64
    let m_f64 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let _ = m_f64.add(&m_f64).unwrap();
    let _ = m_f64.multiply(&m_f64).unwrap();
    
    // Works with f32
    let m_f32 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let _ = m_f32.subtract(&m_f32).unwrap();
    
    // Works with integers  
    let m_i32 = Matrix::new(2, 2, vec![1, 2, 3, 4]).unwrap();
    let _ = m_i32.hadamard_product(&m_i32).unwrap();
}
```

## Custom Initialization

Instead of hardcoding different `random()` methods for He/Xavier/etc., pass a closure:

```rust
use stratrix::Matrix;

fn matrix_he_init(rows: usize, cols: usize) -> Matrix<f64> {
    let limit = (6.0 / cols as f64).sqrt();
    Matrix::random(rows, cols, || {
        rand::thread_rng().gen_range(-limit..limit)
    })
}

fn matrix_xavier_init(rows: usize, cols: usize) -> Matrix<f64> {
    let limit = (6.0 / (rows + cols) as f64).sqrt();
    Matrix::random(rows, cols, || {
        rand::thread_rng().gen_range(-limit..limit)
    })
}
```

## Controlling Block Size

Block size for tiled strategies defaults to 32 and can be customized when creating the matrix:

```rust
use stratrix::{Matrix, Strategy, MatrixConfig};

fn main() {
    // Use custom block size of 64 with TiledParallel strategy
    let config = MatrixConfig::new(Strategy::TiledParallel, 64);
    let matrix = Matrix::random_with_config(1000, 1000, || rand::random::<f64>(), config);
    
    let result = matrix.multiply(&matrix).unwrap();
}
```

## Strategy Selection

Choose based on your use case:
- `Naive`: Simple nested loops, good for basis analysis
- `NaiveParallel`: Parallelized simple nested loops
- `Tiled`: Cache-optimized single-threaded (respects CPU cache lines)
- `TiledParallel`: Cache-optimized with parallelization
