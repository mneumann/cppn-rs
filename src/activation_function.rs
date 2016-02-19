use std::fmt::Debug;

pub trait ActivationFunction: Clone + Debug {
    fn formula(&self) -> &'static str;
    fn calculate(&self, f: f64) -> f64;
}
