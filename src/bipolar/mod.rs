use self::closed01bipolar::Closed01Bipolar;

pub mod closed01bipolar;

pub trait ActivationFunction {
    fn formula() -> &'static str;
    fn calculate(x: f64) -> Closed01Bipolar<f64>;
}

pub struct Linear;
pub struct Gaussian;
pub struct Sigmoid;
pub struct Sine;

impl ActivationFunction for Linear {
    fn formula() -> &'static str {
        "y = max(-1.0, min(1.0, x))"
    }

    fn calculate(x: f64) -> Closed01Bipolar<f64> {
        Closed01Bipolar::new_clipped(x)
    }
}

impl ActivationFunction for Gaussian {
    fn formula() -> &'static str {
        "y = 2.0 * exp(-(x * 2.5)^2.0) - 1.0"
    }

    fn calculate(x: f64) -> Closed01Bipolar<f64> {
        Closed01Bipolar::new(2.0 * (-(x * 2.5).powi(2)).exp() - 1.0)
    }
}

impl ActivationFunction for Sigmoid {
    fn formula() -> &'static str {
        "y = 2.0 / (1.0 + exp(-4.9 * x)) - 1.0"
    }

    fn calculate(x: f64) -> Closed01Bipolar<f64> {
        Closed01Bipolar::new((2.0 / (1.0 + (-4.9 * x).exp())) - 1.0)
    }
}

impl ActivationFunction for Sine {
    fn formula() -> &'static str {
        "y = sin(2.0 * x)"
    }

    fn calculate(x: f64) -> Closed01Bipolar<f64> {
        Closed01Bipolar::new(2.0 * x.sin())
    }
}
