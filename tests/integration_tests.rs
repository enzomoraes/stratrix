mod public_api {
    use stratrix::{Matrix, Strategy, MatrixConfig};

    macro_rules! test_all_strategies {
        ($name:ident, $test_fn:path) => {
            #[test]
            fn $name() {
                for strategy in &[
                    Strategy::Naive,
                    Strategy::NaiveParallel,
                    Strategy::Tiled,
                    Strategy::TiledParallel,
                ] {
                    $test_fn(*strategy);
                }
            }
        };
    }

    fn test_multiply_basic(strategy: Strategy) {
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

    fn test_add_basic(strategy: Strategy) {
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

    fn test_subtract_basic(strategy: Strategy) {
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

    fn test_hadamard_basic(strategy: Strategy) {
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

    test_all_strategies!(multiply_all_strategies, test_multiply_basic);
    test_all_strategies!(add_all_strategies, test_add_basic);
    test_all_strategies!(subtract_all_strategies, test_subtract_basic);
    test_all_strategies!(hadamard_all_strategies, test_hadamard_basic);

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
    fn transpose_matrix() {
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
    fn apply_function() {
        let a = vec![1.0, 10.0, 4.0, 9.0];
        let matrix_a: Matrix<f64> = Matrix::new(2, 2, a).unwrap();
        let result = matrix_a.apply_function(&|x| x * 2.0);

        assert_eq!(result.rows(), 2);
        assert_eq!(result.cols(), 2);
        assert_eq!(result.data()[0], 2.0);
        assert_eq!(result.data()[1], 20.0);
        assert_eq!(result.data()[2], 8.0);
        assert_eq!(result.data()[3], 18.0);
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
}

mod edge_cases {
    use stratrix::Matrix;

    #[test]
    fn single_element_matrix() {
        let matrix: Matrix<f64> = Matrix::new(1, 1, vec![5.0]).unwrap();
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 1);
        assert_eq!(matrix.data()[0], 5.0);
    }

    #[test]
    fn single_row_matrix() {
        let matrix: Matrix<f64> = Matrix::new(1, 3, vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 3);
    }

    #[test]
    fn single_column_matrix() {
        let matrix: Matrix<f64> = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.cols(), 1);
    }

    #[test]
    fn multiply_with_zeros() {
        let a = vec![0.0, 0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let matrix_a: Matrix<f64> = Matrix::new(2, 2, a).unwrap();
        let matrix_b: Matrix<f64> = Matrix::new(2, 2, b).unwrap();
        let result = matrix_a.multiply(&matrix_b).unwrap();

        for val in result.data().iter() {
            assert_eq!(*val, 0.0);
        }
    }

    #[test]
    fn add_negative_numbers() {
        let a = vec![-1.0, -2.0, -3.0, -4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let matrix_a: Matrix<f64> = Matrix::new(2, 2, a).unwrap();
        let matrix_b: Matrix<f64> = Matrix::new(2, 2, b).unwrap();
        let result = matrix_a.add(&matrix_b).unwrap();

        for val in result.data().iter() {
            assert_eq!(*val, 0.0);
        }
    }

    #[test]
    fn subtract_negative_numbers() {
        let a = vec![-1.0, -2.0, -3.0, -4.0];
        let b = vec![-1.0, -2.0, -3.0, -4.0];
        let matrix_a: Matrix<f64> = Matrix::new(2, 2, a).unwrap();
        let matrix_b: Matrix<f64> = Matrix::new(2, 2, b).unwrap();
        let result = matrix_a.subtract(&matrix_b).unwrap();

        for val in result.data().iter() {
            assert_eq!(*val, 0.0);
        }
    }

    #[test]
    fn hadamard_with_zeros() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.0, 0.0, 0.0, 0.0];
        let matrix_a: Matrix<f64> = Matrix::new(2, 2, a).unwrap();
        let matrix_b: Matrix<f64> = Matrix::new(2, 2, b).unwrap();
        let result = matrix_a.hadamard_product(&matrix_b).unwrap();

        for val in result.data().iter() {
            assert_eq!(*val, 0.0);
        }
    }

    #[test]
    fn transpose_single_element() {
        let matrix: Matrix<f64> = Matrix::new(1, 1, vec![5.0]).unwrap();
        let transposed = matrix.transpose();
        assert_eq!(transposed.rows(), 1);
        assert_eq!(transposed.cols(), 1);
        assert_eq!(transposed.data()[0], 5.0);
    }

    #[test]
    fn transpose_single_row() {
        let matrix: Matrix<f64> = Matrix::new(1, 3, vec![1.0, 2.0, 3.0]).unwrap();
        let transposed = matrix.transpose();
        assert_eq!(transposed.rows(), 3);
        assert_eq!(transposed.cols(), 1);
        assert_eq!(transposed.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn apply_function_with_negative_values() {
        let a = vec![-2.0, -1.0, 0.0, 1.0];
        let matrix: Matrix<f64> = Matrix::new(2, 2, a).unwrap();
        let result = matrix.apply_function(&|x| x.abs());

        assert_eq!(result.data()[0], 2.0);
        assert_eq!(result.data()[1], 1.0);
        assert_eq!(result.data()[2], 0.0);
        assert_eq!(result.data()[3], 1.0);
    }
}

mod error_handling {
    use stratrix::Matrix;

    #[test]
    fn multiply_incompatible_dimensions() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let matrix_a: Matrix<f64> = Matrix::new(2, 3, a).unwrap();
        let matrix_b: Matrix<f64> = Matrix::new(2, 2, b).unwrap();
        assert!(matrix_a.multiply(&matrix_b).is_err());
    }

    #[test]
    fn add_incompatible_dimensions() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let matrix_a: Matrix<f64> = Matrix::new(2, 1, a).unwrap();
        let matrix_b: Matrix<f64> = Matrix::new(2, 2, b).unwrap();
        assert!(matrix_a.add(&matrix_b).is_err());
    }

    #[test]
    fn subtract_incompatible_dimensions() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let matrix_a: Matrix<f64> = Matrix::new(1, 3, a).unwrap();
        let matrix_b: Matrix<f64> = Matrix::new(1, 2, b).unwrap();
        assert!(matrix_a.subtract(&matrix_b).is_err());
    }

    #[test]
    fn hadamard_incompatible_dimensions() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0];
        let matrix_a: Matrix<f64> = Matrix::new(2, 2, a).unwrap();
        let matrix_b: Matrix<f64> = Matrix::new(1, 3, b).unwrap();
        assert!(matrix_a.hadamard_product(&matrix_b).is_err());
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
    fn invalid_matrix_creation() {
        let result = Matrix::<f64>::new(2, 2, vec![1.0, 2.0, 3.0]);
        assert!(result.is_err());
    }

    #[test]
    fn invalid_data_length() {
        let result = Matrix::<f64>::new(2, 2, vec![1.0, 2.0, 3.0]);
        assert!(result.is_err());
    }
}

mod type_generics {
    use stratrix::{Matrix, Strategy, MatrixConfig};

    #[test]
    fn add_with_f32() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let matrix_a: Matrix<f32> = Matrix::new(2, 2, a).unwrap();
        let matrix_b: Matrix<f32> = Matrix::new(2, 2, b).unwrap();
        let result = matrix_a.add(&matrix_b).unwrap();

        assert_eq!(result.rows(), 2);
        assert_eq!(result.cols(), 2);
        assert_eq!(result.data()[0], 6.0);
    }

    #[test]
    fn subtract_with_i32() {
        let a = vec![10i32, 20, 30, 40];
        let b = vec![1i32, 2, 3, 4];
        let matrix_a: Matrix<i32> = Matrix::new(2, 2, a).unwrap();
        let matrix_b: Matrix<i32> = Matrix::new(2, 2, b).unwrap();
        let result = matrix_a.subtract(&matrix_b).unwrap();

        assert_eq!(result.rows(), 2);
        assert_eq!(result.cols(), 2);
        assert_eq!(result.data()[0], 9);
    }

    #[test]
    fn hadamard_with_f32() {
        let a = vec![2.0f32, 3.0, 4.0, 5.0];
        let b = vec![2.0f32, 2.0, 2.0, 2.0];
        let matrix_a: Matrix<f32> = Matrix::new(2, 2, a).unwrap();
        let matrix_b: Matrix<f32> = Matrix::new(2, 2, b).unwrap();
        let result = matrix_a.hadamard_product(&matrix_b).unwrap();

        assert_eq!(result.data()[0], 4.0);
        assert_eq!(result.data()[1], 6.0);
    }

    #[test]
    fn multiply_with_i32_strategy() {
        let a = vec![1i32, 2, 3, 4];
        let b = vec![5i32, 6, 7, 8];
        let config = MatrixConfig::new(Strategy::Naive, 32);
        let matrix_a: Matrix<i32> = Matrix::new_with_config(2, 2, a, config).unwrap();
        let matrix_b: Matrix<i32> = Matrix::new(2, 2, b).unwrap();
        let result = matrix_a.multiply(&matrix_b).unwrap();

        assert_eq!(result.rows(), 2);
        assert_eq!(result.cols(), 2);
    }
}
