#[cfg(test)]
extern crate rand;
extern crate acyclic_network;

use std::fmt::Debug;

pub trait ActivationFunction: Clone + Debug {
    fn formula(&self) -> &'static str;
    fn calculate(&self, f: f64) -> f64;
}

pub mod bipolar;
pub mod cppn;
pub mod position;
pub mod substrate;
