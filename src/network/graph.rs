use std::{
    collections::HashMap,
    io::{self, Write},
};

use rand::{RngExt, SeedableRng, rngs::StdRng};

#[derive(Debug, Clone)]
pub struct Graph {
    pub adjacency: HashMap<u32, Vec<u32>>,
    pub seed: Option<usize>,
}

impl Graph {
    fn new(n: usize) -> Self {
        return Self {
            adjacency: HashMap::with_capacity(n),
            seed: None,
        };
    }

    fn add_edge(&mut self, u: usize, v: usize) {
        let u32 = u as u32;
        let v32 = v as u32;

        if let Some(x) = self.adjacency.get_mut(&u32) {
            x.push(v32);
        } else {
            self.adjacency.insert(u32, Vec::from(&[v32]));
        }

        if let Some(x) = self.adjacency.get_mut(&v32) {
            x.push(u32);
        } else {
            self.adjacency.insert(v32, Vec::from(&[u32]));
        }
    }

    pub fn gen_ring(n: usize) -> Self {
        let mut res = Self::new(n);

        for i in 0..n {
            let prev = if i == 0 { n - 1 } else { i - 1 };
            let next = if i + 1 >= n { 0 } else { i + 1 };
            res.adjacency
                .insert(i as u32, Vec::from(&[prev as u32, next as u32]));
        }
        res
    }

    fn gen_random(n: usize, p: f32, seed: usize) -> (bool, Self) {
        let mut res = Self::new(n);

        let mut rng = StdRng::seed_from_u64(seed as u64);

        for u in 0..n {
            for v in (u + 1)..n {
                if rng.random::<f32>() > p {
                    res.add_edge(u, v);
                }
            }
        }
        res.seed = Some(seed);
        (res.adjacency.len() == n, res)
    }

    fn generate_p(n: f32, k: f32) -> f32 {
        1.0 - (k / (n - 1.0))
    }

    pub fn gen_random_with_start_seed(n: usize, seed: usize) -> (usize, Self) {
        let mut iterations = 0;

        let average_degree = (n.ilog2() as f32 + 1.0).floor();
        let p = Self::generate_p(n as f32, average_degree);

        loop {
            let curr_seed = seed + iterations;
            print!("\rTrying seed: {:<20}", curr_seed);
            io::stdout().flush().unwrap();
            let (correct_size, res) = Self::gen_random(n, p, curr_seed);
            if correct_size {
                println!("\nGraph with {n} nodes created using seed {curr_seed}");
                return (curr_seed, res);
            } else {
                iterations += 1;
            }
        }
    }

    pub fn get_average_degree(&self) -> f64 {
        let mut degrees = 0;
        let mut count = 0;
        for x in self.adjacency.keys() {
            count += 1;
            let r = self.adjacency.get(x).unwrap();
            degrees += r.len();
        }
        degrees as f64 / count as f64
    }
}
