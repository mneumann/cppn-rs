use std::fmt::Debug;

pub trait ActivationFunction: Clone + Debug + Send + Sized {
    fn formula_gnuplot(&self, x: &str) -> String;

    fn calculate(&self, x: f64) -> f64;
}

#[inline(always)]
fn bipolar_debug_check(x: f64) -> f64 {
    debug_assert!(x >= -1.0 && x <= 1.0);
    x
}

/// Clips the value of `x` into the range [-1, 1].
fn bipolar_clip(x: f64) -> f64 {
    if x > 1.0 {
        1.0
    } else if x < -1.0 {
        -1.0
    } else {
        x
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GeometricActivationFunction {
    Linear,
    LinearBipolarClipped,
    Absolute,
    Gaussian,
    BipolarGaussian,
    BipolarSigmoid,
    Sine,
    Cosine,
}

impl ActivationFunction for GeometricActivationFunction {
    fn calculate(&self, x: f64) -> f64 {
        match *self {
            GeometricActivationFunction::Linear => x,
            GeometricActivationFunction::LinearBipolarClipped => bipolar_debug_check(bipolar_clip(x)),
            GeometricActivationFunction::Absolute => x.abs(),
            GeometricActivationFunction::Gaussian => {
                (-(x * 2.5).powi(2)).exp()
            }
            GeometricActivationFunction::BipolarGaussian => {
                bipolar_debug_check(2.0 * (-(x * 2.5).powi(2)).exp() - 1.0)
            }
            GeometricActivationFunction::BipolarSigmoid => {
                bipolar_debug_check((2.0 / (1.0 + (-4.9 * x).exp())) - 1.0)
            }
            GeometricActivationFunction::Sine => bipolar_debug_check(x.sin()),
            GeometricActivationFunction::Cosine => bipolar_debug_check(x.cos()),
        }
    }

    fn formula_gnuplot(&self, x: &str) -> String {
        match *self {
            GeometricActivationFunction::Linear => format!("{}", x),
            GeometricActivationFunction::LinearBipolarClipped => format!("max(-1.0, min(1.0, {}))", x),
            GeometricActivationFunction::Absolute => format!("abs({})", x),
            GeometricActivationFunction::Gaussian => format!("(exp(-(({}) * 2.5)**2.0)", x),
            GeometricActivationFunction::BipolarGaussian => format!("2.0 * exp(-(({}) * 2.5)**2.0) - 1.0", x),
            GeometricActivationFunction::BipolarSigmoid => format!("2.0 / (1.0 + exp(-4.9 * ({}))) - 1.0", x),
            GeometricActivationFunction::Sine => format!("sin({})", x),
            GeometricActivationFunction::Cosine => format!("cos({})", x),
        }
    }
}


#[test]
fn test_bipolar_linear() {
    assert_eq!(0.0, GeometricActivationFunction::LinearBipolarClipped.calculate(0.0));
    assert_eq!(1.0, GeometricActivationFunction::LinearBipolarClipped.calculate(1.0));
    assert_eq!(-1.0, GeometricActivationFunction::LinearBipolarClipped.calculate(-1.0));
    assert_eq!(0.5, GeometricActivationFunction::LinearBipolarClipped.calculate(0.5));
    assert_eq!(-0.5, GeometricActivationFunction::LinearBipolarClipped.calculate(-0.5));
    assert_eq!(1.0, GeometricActivationFunction::LinearBipolarClipped.calculate(1.1));
    assert_eq!(-1.0, GeometricActivationFunction::LinearBipolarClipped.calculate(-1.1));
}
