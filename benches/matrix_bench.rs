use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use stratrix::{Matrix, MatrixConfig, Strategy};

fn make_matrix(size: usize, strategy: Strategy) -> Matrix<f64> {
    let data = vec![1.0; size * size];
    let config = MatrixConfig::new(strategy, 32);
    Matrix::new_with_config(size, size, data, config).unwrap()
}

fn bench_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiply");
    group.sample_size(10);

    for &size in &[128usize, 256] {
        for &(name, strategy) in &[
            ("Naive", Strategy::Naive),
            ("NaiveParallel", Strategy::NaiveParallel),
            ("Tiled", Strategy::Tiled),
            ("TiledParallel", Strategy::TiledParallel),
        ] {
            group.bench_with_input(
                BenchmarkId::new(name, size),
                &(size, strategy),
                |b, &(size, strategy)| {
                    let a = black_box(make_matrix(size, strategy));
                    let b_mat = black_box(make_matrix(size, Strategy::Naive));
                    b.iter(|| a.multiply(&b_mat).unwrap());
                },
            );
        }
    }
    group.finish();
}

fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("add");
    group.sample_size(10);

    for &size in &[256usize, 512, 1000] {
        for &(name, strategy) in &[
            ("Naive", Strategy::Naive),
            ("NaiveParallel", Strategy::NaiveParallel),
            ("Tiled", Strategy::Tiled),
            ("TiledParallel", Strategy::TiledParallel),
        ] {
            group.bench_with_input(
                BenchmarkId::new(name, size),
                &(size, strategy),
                |b, &(size, strategy)| {
                    let a = black_box(make_matrix(size, strategy));
                    let b_mat = black_box(make_matrix(size, Strategy::Naive));
                    b.iter(|| a.add(&b_mat).unwrap());
                },
            );
        }
    }
    group.finish();
}

fn bench_subtract(c: &mut Criterion) {
    let mut group = c.benchmark_group("subtract");
    group.sample_size(10);

    for &size in &[256usize, 512, 1000] {
        for &(name, strategy) in &[
            ("Naive", Strategy::Naive),
            ("NaiveParallel", Strategy::NaiveParallel),
            ("Tiled", Strategy::Tiled),
            ("TiledParallel", Strategy::TiledParallel),
        ] {
            group.bench_with_input(
                BenchmarkId::new(name, size),
                &(size, strategy),
                |b, &(size, strategy)| {
                    let a = black_box(make_matrix(size, strategy));
                    let b_mat = black_box(make_matrix(size, Strategy::Naive));
                    b.iter(|| a.subtract(&b_mat).unwrap());
                },
            );
        }
    }
    group.finish();
}

fn bench_hadamard(c: &mut Criterion) {
    let mut group = c.benchmark_group("hadamard");
    group.sample_size(10);

    for &size in &[256usize, 512, 1000] {
        for &(name, strategy) in &[
            ("Naive", Strategy::Naive),
            ("NaiveParallel", Strategy::NaiveParallel),
            ("Tiled", Strategy::Tiled),
            ("TiledParallel", Strategy::TiledParallel),
        ] {
            group.bench_with_input(
                BenchmarkId::new(name, size),
                &(size, strategy),
                |b, &(size, strategy)| {
                    let a = black_box(make_matrix(size, strategy));
                    let b_mat = black_box(make_matrix(size, Strategy::Naive));
                    b.iter(|| a.hadamard_product(&b_mat).unwrap());
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_multiply, bench_add, bench_subtract, bench_hadamard);
criterion_main!(benches);
