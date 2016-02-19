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

#[derive(Debug, Clone, Copy)]
pub enum BipolarActivationFunction {
    Identity,
    Linear,
    Gaussian,
    Sigmoid,
    Sine,
}

impl ActivationFunction for BipolarActivationFunction {
    fn calculate(&self, x: f64) -> f64 {
        match *self {
            BipolarActivationFunction::Identity => x,
            BipolarActivationFunction::Linear => bipolar_debug_check(bipolar_clip(x)),
            BipolarActivationFunction::Gaussian => {
                bipolar_debug_check(2.0 * (-(x * 2.5).powi(2)).exp() - 1.0)
            }
            BipolarActivationFunction::Sigmoid => {
                bipolar_debug_check((2.0 / (1.0 + (-4.9 * x).exp())) - 1.0)
            }
            BipolarActivationFunction::Sine => bipolar_debug_check((2.0 * x).sin()),
        }
    }

    fn formula(&self) -> &'static str {
        match *self {
            BipolarActivationFunction::Identity => "y = x",
            BipolarActivationFunction::Linear => "y = max(-1.0, min(1.0, x))",
            BipolarActivationFunction::Gaussian => "y = 2.0 * exp(-(x * 2.5)^2.0) - 1.0",
            BipolarActivationFunction::Sigmoid => "y = 2.0 / (1.0 + exp(-4.9 * x)) - 1.0",
            BipolarActivationFunction::Sine => "y = sin(2.0 * x)",
        }
    }
}


#[test]
fn test_bipolar_linear() {
    assert_eq!(0.0, BipolarActivationFunction::Linear.calculate(0.0));
    assert_eq!(1.0, BipolarActivationFunction::Linear.calculate(1.0));
    assert_eq!(-1.0, BipolarActivationFunction::Linear.calculate(-1.0));
    assert_eq!(0.5, BipolarActivationFunction::Linear.calculate(0.5));
    assert_eq!(-0.5, BipolarActivationFunction::Linear.calculate(-0.5));
    assert_eq!(1.0, BipolarActivationFunction::Linear.calculate(1.1));
    assert_eq!(-1.0, BipolarActivationFunction::Linear.calculate(-1.1));
}
