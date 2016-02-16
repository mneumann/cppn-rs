pub trait ActivationFunction {
    fn formula() -> &'static str;
    fn calculate(x: f64) -> f64;
}

pub mod bipolar;
pub mod closed01bipolar;
