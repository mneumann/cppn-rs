use self::closed01bipolar::Closed01Bipolar;

pub mod closed01bipolar;

// make generic over the x and y types.
trait ActivationFunction {
    fn formula(&self) -> String;
    fn calculate(&self, x: f64) -> Closed01Bipolar<f64>;
}

pub struct Linear;

impl ActivationFunction for Linear {
    fn formula(&self) -> String {
        "y = max(-1.0, min(1.0, x))".to_owned()
    }

    fn calculate(&self, x: f64) -> Closed01Bipolar<f64> {
        Closed01Bipolar::new_clipped(x)
    }
}

pub struct Sigmoid;

impl ActivationFunction for Sigmoid {
    fn formula(&self) -> String {
        "y = 2.0 / (1.0 + exp(-4.9 * x)) - 1.0".to_owned()
    }

    fn calculate(&self, x: f64) -> Closed01Bipolar<f64> {
        Closed01Bipolar::new((2.0 / (1.0 + (-4.9 * x).exp())) - 1.0)
    }
}

pub struct Sine;

impl ActivationFunction for Sine {
    fn formula(&self) -> String {
        "y = sin(2.0 * x)".to_owned()
    }

    fn calculate(&self, x: f64) -> Closed01Bipolar<f64> {
        Closed01Bipolar::new(2.0 * x.sin())
    }
}

pub struct Gaussian;

impl ActivationFunction for Gaussian {
    fn formula(&self) -> String {
        "y = 2.0 * exp(-(x * 2.5)^2.0) - 1.0".to_owned()
    }

    fn calculate(&self, x: f64) -> Closed01Bipolar<f64> {
        Closed01Bipolar::new(2.0 * (-(x*2.5).powi(2)).exp() - 1.0)
    }
}
