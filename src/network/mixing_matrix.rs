use std::collections::{HashMap, HashSet};

use crate::network::graph::Graph;

#[derive(Debug)]
pub struct MixingMatrix {
    init: HashMap<u32, f64>,
    current: HashMap<u32, f64>,
}

impl MixingMatrix {
    pub fn new(node_id: u32, graph: &Graph) -> Self {
        let node_adjacent = graph.adjacency.get(&node_id).unwrap();

        let mut init = HashMap::with_capacity(node_adjacent.len());
        let mut current = HashMap::with_capacity(node_adjacent.len());

        for n in node_adjacent {
            let neighbor_adjacent = graph.adjacency.get(n).unwrap().len();
            let w = 1.0 / (1.0 + f64::max(node_adjacent.len() as f64, neighbor_adjacent as f64));
            init.insert(*n, w);
            current.insert(*n, w);
        }

        Self { init, current }
    }

    pub fn reset(&mut self) {
        self.current = self.init.clone();
    }

    pub fn remove(&mut self, id: u32) {
        self.current.remove(&id);
    }

    pub fn remove_batch<T: IntoIterator<Item = u32>>(&mut self, ids: T) {
        for id in ids {
            self.current.remove(&id);
        }
    }

    pub fn filter<T: IntoIterator<Item = u32>>(&mut self, ids: T) {
        let set: HashSet<u32> = ids.into_iter().collect();
        let keys_to_remove: Vec<u32> = self
            .current
            .keys()
            .filter(|id| !set.contains(id))
            .cloned()
            .collect();
        self.remove_batch(keys_to_remove);
    }

    pub fn sorted_vec(&self) -> Vec<(u32, f64)> {
        let mut v: Vec<(u32, f64)> = self.current.iter().map(|(&k, &v)| (k, v)).collect();
        v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        v
    }

    pub fn init_node_weight() -> f64 {
        1.0
    }

    pub fn get_by_id(&self, id: &u32) -> f64 {
        *self
            .current
            .get(id)
            .expect("no element with id exists in mixing matrix")
    }
}
