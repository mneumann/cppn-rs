use super::ActivationFunction;

#[inline(always)]
fn bipolar_debug_check(f: f64) -> f64 {
    debug_assert!(f >= -1.0 && f <= 1.0);
    f
}

/// Clips the value of `f` into the range [-1, 1].
fn bipolar_clip(f: f64) -> f64 {
    if f > 1.0 {
        1.0
    } else if f < -1.0 {
        -1.0
    } else {
        f
    }
}

pub struct Linear;
pub struct Gaussian;
pub struct Sigmoid;
pub struct Sine;

impl ActivationFunction for Linear {
    fn formula(&self) -> &'static str {
        "y = max(-1.0, min(1.0, x))"
    }

    fn calculate(&self, x: f64) -> f64 {
        bipolar_debug_check(bipolar_clip(x))
    }
}

impl ActivationFunction for Gaussian {
    fn formula(&self) -> &'static str {
        "y = 2.0 * exp(-(x * 2.5)^2.0) - 1.0"
    }

    fn calculate(&self, x: f64) -> f64 {
        bipolar_debug_check(2.0 * (-(x * 2.5).powi(2)).exp() - 1.0)
    }
}

impl ActivationFunction for Sigmoid {
    fn formula(&self) -> &'static str {
        "y = 2.0 / (1.0 + exp(-4.9 * x)) - 1.0"
    }

    fn calculate(&self, x: f64) -> f64 {
        bipolar_debug_check((2.0 / (1.0 + (-4.9 * x).exp())) - 1.0)
    }
}

impl ActivationFunction for Sine {
    fn formula(&self) -> &'static str {
        "y = sin(2.0 * x)"
    }

    fn calculate(&self, x: f64) -> f64 {
        bipolar_debug_check((2.0 * x).sin())
    }
}

#[test]
fn test_linear() {
    assert_eq!(0.0, Linear.calculate(0.0));
    assert_eq!(1.0, Linear.calculate(1.0));
    assert_eq!(-1.0, Linear.calculate(-1.0));
    assert_eq!(0.5, Linear.calculate(0.5));
    assert_eq!(-0.5, Linear.calculate(-0.5));
    assert_eq!(1.0, Linear.calculate(1.1));
    assert_eq!(-1.0, Linear.calculate(-1.1));
}
