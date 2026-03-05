use std::collections::HashMap;

#[derive(Debug)]
pub struct ReputationTable {
    table: HashMap<(u32, u32), f64>
}

impl ReputationTable {
    pub fn new() -> Self {
        ReputationTable { table: HashMap::new() }
    }

    pub fn put(&mut self, (node, neighbor): (u32, u32), score: f64) {
        self.table.insert((node, neighbor), score);
    }

    pub fn get(&self, key: &(u32, u32)) -> f64 {
        self.table.get(key).expect("tried to get invalid reputation score").to_owned()
    }
}
