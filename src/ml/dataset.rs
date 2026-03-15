use std::collections::HashMap;

use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use rand_distr::{Distribution, multi::Dirichlet};
use tch::{
    Device, Tensor,
    vision::{dataset::Dataset, mnist},
};

use crate::config::config::CONFIG;

pub struct DataSet {
    splits: Vec<(Tensor, Tensor)>,
    loop_test_set: (Tensor, Tensor),
    final_test_set: (Tensor, Tensor),
}

fn generate_test_set(data: &Dataset, k: u32) -> (Tensor, Tensor) {
    let full_test_x = data
        .test_images
        .view([data.test_images.size()[0], 1, 28, 28])
        .to(CONFIG.device);
    let full_test_y = data.test_labels.to(CONFIG.device);
    let n_test = full_test_x.size()[0];

    tch::manual_seed(CONFIG.seed as i64);
    let idx = Tensor::randperm(n_test, (tch::Kind::Int64, CONFIG.device)).narrow(0, 0, k.into());
    let test_images = full_test_x.index_select(0, &idx);
    let test_labels = full_test_y.index_select(0, &idx);
    (test_images, test_labels)
}

impl DataSet {
    pub fn from_path(n: usize, alpha: f64, seed: u64) -> Self {
        let data =
            mnist::load_dir(CONFIG.data.path.clone()).expect("invalid path to mnist data set");

        let loop_test_set = generate_test_set(&data, CONFIG.training.loop_test_size);
        let final_test_set = generate_test_set(&data, CONFIG.training.final_test_size);
        let train_images = data.train_images;
        let train_labels = data.train_labels;

        let num_classes = 10;
        let mut class_indices: HashMap<i64, Vec<i64>> = HashMap::new();

        for i in 0..train_labels.size()[0] {
            let label = train_labels.int64_value(&[i]);
            class_indices.entry(label).or_default().push(i);
        }

        let mut node_indices: Vec<Vec<i64>> = vec![vec![]; n];
        let mut rng = StdRng::seed_from_u64(seed as u64);

        for c in 0..num_classes {
            let indices = &mut class_indices.get_mut(&c).unwrap();
            indices.shuffle(&mut rng);

            let dir = Dirichlet::new(&vec![alpha; n]).unwrap();
            let proportions = dir.sample(&mut rng);

            let class_size = indices.len();
            let mut start = 0;

            for (node_idx, p) in proportions.iter().enumerate() {
                let count = ((p * class_size as f64).round() as usize).min(class_size - start);
                let end = start + count;
                if end > start {
                    node_indices[node_idx].extend_from_slice(&indices[start..end]);
                }
                start = end;
            }
        }
        Self {
            loop_test_set,
            final_test_set,
            splits: node_indices
                .iter()
                .map(|idx| {
                    let idx_tensor = Tensor::from_slice(&idx);
                    let x = train_images
                        .index_select(0, &idx_tensor)
                        .view([idx.len() as i64, 1, 28, 28])
                        .to(CONFIG.device);
                    let y = train_labels.index_select(0, &idx_tensor).to(CONFIG.device);
                    (x, y)
                })
                .collect(),
        }
    }

    pub fn shuffle(&mut self, rng: &mut StdRng) {
        self.splits.shuffle(rng);
    }

    pub fn pop(&mut self) -> (Tensor, Tensor) {
        self.splits
            .pop()
            .expect("split count is equal to node count")
    }

    pub fn loop_test_set(&self) -> (Tensor, Tensor) {
        (
            self.loop_test_set.0.shallow_clone(),
            self.loop_test_set.1.shallow_clone(),
        )
    }

    pub fn final_test_set(&self) -> (Tensor, Tensor) {
        (
            self.final_test_set.0.shallow_clone(),
            self.final_test_set.1.shallow_clone(),
        )
    }

    pub fn check_label_distribution(&self, n: usize) {
        let x = Self::from_path(n, CONFIG.data.dirichlet_alpha as f64, 123);

        for (id, (_x, y)) in x.splits.iter().enumerate() {
            let mut label_vec = vec![0; 10];
            for i in 0..y.size()[0] {
                label_vec[y.int64_value(&[i]) as usize] += 1;
            }
            let y_len = y.size()[0];

            let percentage = |x: i64| -> f64 {
                return x as f64 / y_len as f64;
            };

            println!(
                "split {:<2} distribution:\nclass 0: {:<2}, class 1: {:<2}\nclass 2: {:<2}, class 3: {:<2}\nclass 4: {:<2}, class 5: {:<2}\nclass 6: {:<2}, class 7: {:<2}\n class 8: {:<2}, class 9: {:<2}",
                id,
                percentage(label_vec[0]),
                percentage(label_vec[1]),
                percentage(label_vec[2]),
                percentage(label_vec[3]),
                percentage(label_vec[4]),
                percentage(label_vec[5]),
                percentage(label_vec[6]),
                percentage(label_vec[7]),
                percentage(label_vec[8]),
                percentage(label_vec[9])
            );
        }
    }
}

#[derive(Debug)]
pub struct LocalDataSet {
    pub train_x: Tensor,
    pub test_x: Tensor,
    pub train_y: Tensor,
    pub test_y: Tensor,
    pub batch_size: usize,
    pub current_idx: usize,
    pub device: Device,
}

impl LocalDataSet {
    pub fn new(x: Tensor, y: Tensor, batch_size: usize, test_ratio: f64, device: Device) -> Self {
        let n_samples = x.size()[0] as usize;
        let test_size = (n_samples as f64 * test_ratio).round() as usize;
        let train_size = n_samples - test_size;

        let indices = Tensor::randperm(n_samples as i64, (tch::Kind::Int64, device));
        let x_shuffled = x.index_select(0, &indices);
        let y_shuffled = y.index_select(0, &indices);

        let train_x = x_shuffled.narrow(0, 0, train_size as i64);
        let train_y = y_shuffled.narrow(0, 0, train_size as i64);
        let test_x = x_shuffled.narrow(0, train_size as i64, test_size as i64);
        let test_y = y_shuffled.narrow(0, train_size as i64, test_size as i64);

        Self {
            train_x,
            test_x,
            train_y,
            test_y,
            batch_size,
            current_idx: 0,
            device,
        }
    }

    pub fn train_len(&self) -> usize {
        self.train_x.size()[0] as usize
    }

    pub fn test_len(&self) -> usize {
        self.test_x.size()[0] as usize
    }

    pub fn shuffle_train(&mut self) {
        let indices = Tensor::randperm(self.train_len() as i64, (tch::Kind::Int64, self.device));
        self.train_x = self.train_x.index_select(0, &indices);
        self.train_y = self.train_y.index_select(0, &indices);
        self.current_idx = 0;
    }

    pub fn next_train_batch(&mut self) -> Option<(Tensor, Tensor)> {
        if self.current_idx >= self.train_len() {
            self.reset();
            self.shuffle_train();
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.train_len());
        let batch_x = self.train_x.narrow(
            0,
            self.current_idx as i64,
            (end_idx - self.current_idx) as i64,
        );
        let batch_y = self.train_y.narrow(
            0,
            self.current_idx as i64,
            (end_idx - self.current_idx) as i64,
        );
        self.current_idx = end_idx;
        Some((batch_x, batch_y))
    }

    pub fn reset(&mut self) {
        self.current_idx = 0;
    }

    pub fn get_test_data(&self) -> (Tensor, Tensor) {
        (self.test_x.shallow_clone(), self.test_y.shallow_clone())
    }
}
