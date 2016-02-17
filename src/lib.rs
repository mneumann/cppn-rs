#[cfg(test)]
extern crate rand;
extern crate acyclic_network;

pub trait ActivationFunction {
    fn formula(&self) -> &'static str;
    fn calculate(&self, x: f64) -> f64;
}

pub struct Identity;

impl ActivationFunction for Identity {
    fn formula(&self) -> &'static str {
        "y = x"
    }

    fn calculate(&self, x: f64) -> f64 {
        x
    }
}

pub mod bipolar;
pub mod closed01bipolar;
pub mod cppn;
pub mod position;
pub mod substrate;
