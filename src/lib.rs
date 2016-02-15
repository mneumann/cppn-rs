use std::convert::Into;

pub trait ActivationFunction {
    type Output: Into<f64>;
    fn formula() -> &'static str;
    fn calculate(x: f64) -> Self::Output;
}

pub mod bipolar;
