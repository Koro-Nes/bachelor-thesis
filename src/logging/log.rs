use std::sync::{Arc, Mutex};

use once_cell::sync::Lazy;

pub struct Logger {}

pub static LOG: Lazy<Arc<Mutex<Logger>>> = Lazy::new(|| Arc::new(Mutex::new(Logger {})));
