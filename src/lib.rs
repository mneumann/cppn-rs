pub trait ActivationFunction {
    fn formula() -> &'static str;
    fn calculate(x: f64) -> f64;
}

pub struct Identity;

impl ActivationFunction for Identity {
    fn formula() -> &'static str {
        "y = x"
    }

    fn calculate(x: f64) -> f64 {
        x
    }
}

pub mod bipolar;
pub mod closed01bipolar;
