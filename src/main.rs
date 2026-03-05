use repository::{
    args, config::{config::CONFIG, *}, ml::dataset::DataSet, network::graph::Graph, node::node::Node
};

fn main() {
    let n = CONFIG.graph.n as usize;

    let _graph = Graph::gen_ring(n);

    let graph_rand = Graph::gen_random_with_start_seed(n, 5);

    let mut degrees = 0;
    let mut count = 0;
    for x in graph_rand.adjacency.keys() {
        count += 1;
        let r = graph_rand.adjacency.get(x).unwrap();
        degrees += r.len();
    }
    println!("average degree: {}", degrees / count);

    DataSet::check_label_distribution();

    //let x = Node::generate_nodes_from_graph(&graph, args.byzantine_fraction, 0.5, 5);
}
