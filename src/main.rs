mod args;

fn main() {
    let args = args::args::Args::get();
    println!("{:?}", args);
}
