extern crate acyclic_network;
extern crate fixedbitset;
#[cfg(test)]
extern crate rand;

#[cfg(feature = "serde")]
#[macro_use]
extern crate serde;

pub mod activation_function;
pub mod cppn;
pub mod position;
pub mod substrate;
