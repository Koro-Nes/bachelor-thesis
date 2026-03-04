use repository::{args, network::graph::Graph};

fn main() {
    let args = args::args::Args::get();

    let graph = Graph::gen_ring(10);

    let graph_rand = Graph::gen_random_with_start_seed(args.n as usize, 5);

    println!("Graph:\n{:?}", graph_rand);
}
