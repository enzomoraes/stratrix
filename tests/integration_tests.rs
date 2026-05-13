use stratrix::{Matrix, Strategy, MatrixConfig};

fn test_multiply_with_strategy(strategy: Strategy) {
    let a = vec![1.0, 10.0, 4.0, 9.0];
    let b = vec![3.0, 5.0, 5.0, 7.0];
    let config = MatrixConfig::new(strategy, 32);
    let matrix_a: Matrix<f64> = Matrix::new_with_config(2, 2, a, config).unwrap();
    let matrix_b: Matrix<f64> = Matrix::new(2, 2, b).unwrap();
    let multiplied = matrix_a.multiply(&matrix_b).unwrap();

    assert_eq!(multiplied.rows(), 2);
    assert_eq!(multiplied.cols(), 2);
    assert!((multiplied.data()[0] - 53.0).abs() < 1e-10);
    assert!((multiplied.data()[1] - 75.0).abs() < 1e-10);
    assert!((multiplied.data()[2] - 57.0).abs() < 1e-10);
    assert!((multiplied.data()[3] - 83.0).abs() < 1e-10);
}

fn test_add_with_strategy(strategy: Strategy) {
    let a = vec![1.0, 10.0, 4.0, 9.0];
    let b = vec![3.0, 5.0, 5.0, 7.0];
    let config = MatrixConfig::new(strategy, 32);
    let matrix_a: Matrix<f64> = Matrix::new_with_config(2, 2, a, config).unwrap();
    let matrix_b: Matrix<f64> = Matrix::new(2, 2, b).unwrap();
    let added = matrix_a.add(&matrix_b).unwrap();

    assert_eq!(added.rows(), 2);
    assert_eq!(added.cols(), 2);
    assert_eq!(added.data()[0], 4.0);
    assert_eq!(added.data()[1], 15.0);
    assert_eq!(added.data()[2], 9.0);
    assert_eq!(added.data()[3], 16.0);
}

fn test_subtract_with_strategy(strategy: Strategy) {
    let a = vec![1.0, 10.0, 4.0, 9.0];
    let b = vec![3.0, 5.0, 5.0, 7.0];
    let config = MatrixConfig::new(strategy, 32);
    let matrix_a: Matrix<f64> = Matrix::new_with_config(2, 2, a, config).unwrap();
    let matrix_b: Matrix<f64> = Matrix::new(2, 2, b).unwrap();
    let subtracted = matrix_a.subtract(&matrix_b).unwrap();

    assert_eq!(subtracted.rows(), 2);
    assert_eq!(subtracted.cols(), 2);
    assert_eq!(subtracted.data()[0], -2.0);
    assert_eq!(subtracted.data()[1], 5.0);
    assert_eq!(subtracted.data()[2], -1.0);
    assert_eq!(subtracted.data()[3], 2.0);
}

fn test_hadamard_with_strategy(strategy: Strategy) {
    let a = vec![1.0, 10.0, 4.0, 9.0];
    let b = vec![3.0, 5.0, 5.0, 7.0];
    let config = MatrixConfig::new(strategy, 32);
    let matrix_a: Matrix<f64> = Matrix::new_with_config(2, 2, a, config).unwrap();
    let matrix_b: Matrix<f64> = Matrix::new(2, 2, b).unwrap();
    let result = matrix_a.hadamard_product(&matrix_b).unwrap();

    assert_eq!(result.rows(), 2);
    assert_eq!(result.cols(), 2);
    assert_eq!(result.data()[0], 3.0);
    assert_eq!(result.data()[1], 50.0);
    assert_eq!(result.data()[2], 20.0);
    assert_eq!(result.data()[3], 63.0);
}

#[test]
fn matrix_creation() {
    let matrix: Matrix<f64> = Matrix::new(2, 2, vec![1.0, 1.5, 2.0, 2.5]).unwrap();
    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.cols(), 2);
    assert_eq!(matrix.data()[0], 1.0);
    assert_eq!(matrix.data()[1], 1.5);
    assert_eq!(matrix.data()[2], 2.0);
    assert_eq!(matrix.data()[3], 2.5);
}

#[test]
fn matrix_of_zeros() {
    let matrix: Matrix<f64> = Matrix::zeros(2, 2);
    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.cols(), 2);
    for val in matrix.data().iter() {
        assert_eq!(*val, 0.0);
    }
}

#[test]
fn identity_matrix() {
    let matrix: Matrix<f64> = Matrix::identity(3);
    assert_eq!(matrix.rows(), 3);
    assert_eq!(matrix.cols(), 3);
    assert_eq!(matrix.data()[0], 1.0);
    assert_eq!(matrix.data()[4], 1.0);
    assert_eq!(matrix.data()[8], 1.0);
    assert_eq!(matrix.data()[1], 0.0);
}

#[test]
fn multiply_naive() {
    test_multiply_with_strategy(Strategy::Naive);
}

#[test]
fn multiply_naive_parallel() {
    test_multiply_with_strategy(Strategy::NaiveParallel);
}

#[test]
fn multiply_tiled() {
    test_multiply_with_strategy(Strategy::Tiled);
}

#[test]
fn multiply_tiled_parallel() {
    test_multiply_with_strategy(Strategy::TiledParallel);
}

#[test]
fn add_naive() {
    test_add_with_strategy(Strategy::Naive);
}

#[test]
fn add_naive_parallel() {
    test_add_with_strategy(Strategy::NaiveParallel);
}

#[test]
fn add_tiled() {
    test_add_with_strategy(Strategy::Tiled);
}

#[test]
fn add_tiled_parallel() {
    test_add_with_strategy(Strategy::TiledParallel);
}

#[test]
fn subtract_naive() {
    test_subtract_with_strategy(Strategy::Naive);
}

#[test]
fn subtract_naive_parallel() {
    test_subtract_with_strategy(Strategy::NaiveParallel);
}

#[test]
fn subtract_tiled() {
    test_subtract_with_strategy(Strategy::Tiled);
}

#[test]
fn subtract_tiled_parallel() {
    test_subtract_with_strategy(Strategy::TiledParallel);
}

#[test]
fn hadamard_naive() {
    test_hadamard_with_strategy(Strategy::Naive);
}

#[test]
fn hadamard_naive_parallel() {
    test_hadamard_with_strategy(Strategy::NaiveParallel);
}

#[test]
fn hadamard_tiled() {
    test_hadamard_with_strategy(Strategy::Tiled);
}

#[test]
fn hadamard_tiled_parallel() {
    test_hadamard_with_strategy(Strategy::TiledParallel);
}

#[test]
fn transpose_matrix_test() {
    let a = vec![1.0, 10.0, 5.0, 4.0, 9.0, 3.0];
    let matrix_a: Matrix<f64> = Matrix::new(2, 3, a).unwrap();
    let transposed = matrix_a.transpose();

    assert_eq!(transposed.rows(), 3);
    assert_eq!(transposed.cols(), 2);
    assert_eq!(transposed.data()[0], 1.0);
    assert_eq!(transposed.data()[1], 4.0);
    assert_eq!(transposed.data()[2], 10.0);
    assert_eq!(transposed.data()[3], 9.0);
    assert_eq!(transposed.data()[4], 5.0);
    assert_eq!(transposed.data()[5], 3.0);
}

#[test]
fn apply_function_test() {
    let a = vec![1.0, 10.0, 4.0, 9.0];
    let matrix_a: Matrix<f64> = Matrix::new(2, 2, a).unwrap();
    let multiplied = matrix_a.apply_function(&|x| x * 2.0);

    assert_eq!(multiplied.rows(), 2);
    assert_eq!(multiplied.cols(), 2);
    assert_eq!(multiplied.data()[0], 2.0);
    assert_eq!(multiplied.data()[1], 20.0);
    assert_eq!(multiplied.data()[2], 8.0);
    assert_eq!(multiplied.data()[3], 18.0);
}

#[test]
fn matrix_with_integer_type() {
    let a = vec![1i32, 2, 3, 4];
    let matrix_a: Matrix<i32> = Matrix::new(2, 2, a).unwrap();
    assert_eq!(matrix_a.rows(), 2);
    assert_eq!(matrix_a.cols(), 2);
    assert_eq!(matrix_a.data()[0], 1);
}

#[test]
fn matrix_with_f32() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let matrix_a: Matrix<f32> = Matrix::new(2, 2, a).unwrap();
    assert_eq!(matrix_a.rows(), 2);
    let b = vec![5.0f32, 6.0, 7.0, 8.0];
    let config = MatrixConfig::new(Strategy::Naive, 32);
    let matrix_b: Matrix<f32> = Matrix::new_with_config(2, 2, b, config).unwrap();
    let result = matrix_a.multiply(&matrix_b).unwrap();
    assert_eq!(result.rows(), 2);
}

#[test]
fn multiply_dimension_mismatch() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let matrix_a: Matrix<f64> = Matrix::new(2, 3, a).unwrap();
    let matrix_b: Matrix<f64> = Matrix::new(2, 2, b).unwrap();
    let result = matrix_a.multiply(&matrix_b);
    assert!(result.is_err());
}

#[test]
fn add_dimension_mismatch() {
    let a = vec![1.0, 2.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let matrix_a: Matrix<f64> = Matrix::new(2, 1, a).unwrap();
    let matrix_b: Matrix<f64> = Matrix::new(2, 2, b).unwrap();
    let result = matrix_a.add(&matrix_b);
    assert!(result.is_err());
}

#[test]
fn random_with_closure() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = Arc::clone(&counter);

    let matrix: Matrix<f64> = Matrix::random(3, 3, || {
        counter_clone.fetch_add(1, Ordering::SeqCst) as f64
    });

    assert_eq!(matrix.rows(), 3);
    assert_eq!(matrix.cols(), 3);
    assert_eq!(counter.load(Ordering::SeqCst), 9);
}

#[test]
fn with_config_pattern() {
    let data = vec![0.0; 9];
    let config = MatrixConfig::new(Strategy::NaiveParallel, 64);
    let matrix: Matrix<f64> = Matrix::new_with_config(3, 3, data, config).unwrap();

    assert_eq!(matrix.rows(), 3);
    assert_eq!(matrix.cols(), 3);
}

#[test]
fn invalid_data_length() {
    let result = Matrix::<f64>::new(2, 2, vec![1.0, 2.0, 3.0]);
    assert!(result.is_err());
}
