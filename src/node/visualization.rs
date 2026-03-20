use serde::Serialize;

use crate::{logging::io, node::node::Node};

pub struct Visualization {
    pub nodes: Vec<VisualizationNode>
}

impl Visualization {
    pub fn from(nodes: &Vec<Node>) -> Self {
        Self {
            nodes: VisualizationNode::vec_from_vec(&nodes)
        }
    }

    pub fn export_json(self, seed: usize, n: usize, byzantine_fraction: f32, collusion_fraction: f32) {
        let out_path = format!("seed{seed}_n{n}_bf{byzantine_fraction}_cf{collusion_fraction}.json");
        io::export_graph_to_json(&out_path, self)
    }
}

#[derive(Serialize)]
pub struct VisualizationNode {
    id: u32,
    neighbors: Vec<u32>,
    kind: NodeKind,
}

impl VisualizationNode {
    fn from(node: &Node) -> Self {
        Self {
            id: node.id,
            neighbors: node.neighbors.clone(),
            kind: match node.kind {
                super::node::NodeKind::Adversarial { colluding, .. } => {
                    NodeKind::Adversarial { colluding }
                },
                super::node::NodeKind::Benign => NodeKind::Benign
            }
        }
    }

    fn vec_from_vec(nodes: &Vec<Node>) -> Vec<Self> {
        nodes.iter().map(|n| Self::from(n)).collect()
    }

}

#[derive(Serialize)]
enum NodeKind {
    Benign,
    Adversarial {
        colluding: bool,
    }
}