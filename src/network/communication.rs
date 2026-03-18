use std::{
    collections::{HashMap, VecDeque},
    mem::size_of,
};

use tch::{Kind, Tensor};

#[derive(Debug, Clone)]
pub struct Communication {
    id: u32,
    neighbors: Vec<u32>,
    shared_neighbors: Vec<(u32, Vec<u32>)>,
    bytes_sent: u128,
    bytes_received: u128,
}

impl Communication {
    pub fn new(id: u32) -> Self {
        Self {
            id,
            neighbors: Vec::new(),
            shared_neighbors: Vec::new(),
            bytes_received: 0,
            bytes_sent: 0
        }
    }

    pub fn add_neighbor_channels(&mut self, ids: impl Iterator<Item = u32>) {
        for id in ids {
            self.add_neighbor_channel(id);
        }
    }

    fn add_neighbor_channel(&mut self, neighbor_id: u32) {
        self.neighbors.push(neighbor_id);
    }

    pub fn send_neighbor_sharing(&mut self, global_neighbor_channel_list: &mut HashMap<u32, Channel<(u32, u32)>>) {
        let ids: Vec<u32> = self.neighbors.iter().cloned().collect();
        for receiver in &ids {
            let channel = global_neighbor_channel_list.get_mut(receiver).unwrap();
            for shared_id in &ids {
                self.bytes_sent += size_of::<(u32, u32)>() as u128;
                channel.send((self.id, *shared_id));
            }
        }
    }

    pub fn compute_shared_neighbors(&mut self, global_neighbor_channel_list: &mut HashMap<u32, Channel<(u32, u32)>>) {
        let messages = global_neighbor_channel_list.get_mut(&self.id).unwrap();

        for n in &self.neighbors {
            self.shared_neighbors.push((*n, Vec::new()));
        }

        while let Some((neighbor_id, shared_id)) = messages.recv() {
            self.bytes_received += std::mem::size_of::<(u32, u32)>() as u128;
            if self.neighbors.contains(&shared_id) {
                let shared_vec = &mut self.shared_neighbors.iter_mut().find(|n| n.0 == neighbor_id).unwrap().1;
                shared_vec.push(shared_id);
            }
        }
    }

    pub fn send_models(&mut self, model: &Tensor, global_model_channel_list: &mut HashMap<u32, Channel<(u32, Tensor)>>) {
        for n in &self.neighbors {

            self.bytes_sent += (std::mem::size_of::<(u32, Tensor)>() + Self::tensor_nbytes(model)) as u128;

            let channel = global_model_channel_list.get_mut(n).unwrap();
            channel.send((self.id, model.shallow_clone()));
        }
    }

    pub fn receive_models(
        &mut self,
        global_model_channel_list: &mut HashMap<u32, Channel<(u32, Tensor)>>,
    ) -> Vec<(u32, Tensor)> {
        let mut res = Vec::new();
        let channel = global_model_channel_list.get_mut(&self.id).unwrap();
        while let Some(response) = channel.recv() {
            self.bytes_received +=
                (std::mem::size_of::<(u32, Tensor)>() + Self::tensor_nbytes(&response.1)) as u128;
            res.push(response);
        }

        res
    }

    pub fn send_reputation_scores(
        &mut self,
        reputation_scores: &Vec<(u32, f64)>,
        global_reputation_channel_list: &mut HashMap<u32, Channel<(u32, u32, f64)>>,
    ) {
        for (neighbor, shared_neigbors) in &self.shared_neighbors {
            let channel = global_reputation_channel_list.get_mut(neighbor).unwrap();

            for shared in shared_neigbors {
                let shared_score = reputation_scores
                    .iter()
                    .find(|(id, _)| id == shared)
                    .unwrap()
                    .1;

                self.bytes_sent += size_of::<(u32, u32, f64)>() as u128;
                channel.send((self.id, *shared, shared_score));
            }
        }
    }

    pub fn receive_reputation_scores(
        &mut self,
        global_reputation_channel_list: &mut HashMap<u32, Channel<(u32, u32, f64)>>,
    ) -> Vec<(u32, Vec<(u32, f64)>)> {
        let mut res_map: HashMap<u32, Vec<(u32, f64)>> = HashMap::new();
        let channel = global_reputation_channel_list.get_mut(&self.id).unwrap();
        while let Some((sender_id, shared_id, score)) = channel.recv() {
            self.bytes_received += size_of::<(u32, u32, f64)>() as u128;
            res_map.entry(sender_id).or_default().push((shared_id, score));
        }

        let mut res = Vec::new();
        for neighbor in &self.neighbors {
            let scores = res_map.remove(neighbor).unwrap_or_default();
            res.push((*neighbor, scores));
        }

        res
    }


    fn tensor_nbytes(t: &Tensor) -> usize {
        let numel = t.numel(); // total number of elements

        let element_size = match t.kind() {
            Kind::Uint8 | Kind::Int8 | Kind::Bool => 1,
            Kind::Int16 | Kind::Half | Kind::BFloat16 => 2,
            Kind::Int | Kind::Float => 4,
            Kind::Int64 | Kind::Double => 8,
            _ => panic!("Unsupported tensor kind"),
        };

        numel * element_size
    }

    pub fn take_bytes(&mut self) -> (u128, u128) {
        let sent = self.bytes_sent;
        let received = self.bytes_received;
        self.bytes_sent = 0;
        self.bytes_received = 0;
        (sent, received)
    }
}

#[derive(Debug, Clone)]
pub struct Channel<T> {
    queue: VecDeque<T>,
}

impl<T> Channel<T> {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new()
        }
    }

    fn send(&mut self, value: T) {
        self.queue.push_back(value);
    }

    fn recv(&mut self) -> Option<T> {
        self.queue.pop_front()
    }
}
