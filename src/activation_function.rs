use std::fmt::Debug;
use std::f64::consts::PI;

pub trait ActivationFunction: Clone + Debug + Send + Sized + PartialEq + Eq {
    fn formula_gnuplot(&self, x: String) -> String;
    fn name(&self) -> String;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometricActivationFunction {
    Linear,
    LinearBipolarClipped,
    Absolute,
    Gaussian,
    BipolarGaussian,
    BipolarSigmoid,
    Sine,
    Cosine,
    Constant1,
}

impl ActivationFunction for GeometricActivationFunction {
    fn calculate(&self, x: f64) -> f64 {
        match *self {
            GeometricActivationFunction::Linear => x,
            GeometricActivationFunction::LinearBipolarClipped => bipolar_debug_check(bipolar_clip(x)),
            GeometricActivationFunction::Absolute => x.abs(),
            GeometricActivationFunction::Gaussian => {
                (-((x * 2.5).powi(2))).exp()
            }
            GeometricActivationFunction::BipolarGaussian => {
                bipolar_debug_check(2.0 * (-((x * 2.5).powi(2))).exp() - 1.0)
            }
            GeometricActivationFunction::BipolarSigmoid => {
                bipolar_debug_check((2.0 / (1.0 + (-4.9 * x).exp())) - 1.0)
            }
            GeometricActivationFunction::Sine => bipolar_debug_check((2.0*PI*x).sin()),
            GeometricActivationFunction::Cosine => bipolar_debug_check(2.0*PI*x.cos()),
            GeometricActivationFunction::Constant1 => 1.0,
        }
    }

    fn formula_gnuplot(&self, x: String) -> String {
        match *self {
            GeometricActivationFunction::Linear => format!("{}", x),
            GeometricActivationFunction::LinearBipolarClipped => format!("max(-1.0, min(1.0, {}))", x),
            GeometricActivationFunction::Absolute => format!("abs({})", x),
            GeometricActivationFunction::Gaussian => format!("(exp(-((({}) * 2.5)**2.0))", x),
            GeometricActivationFunction::BipolarGaussian => format!("2.0 * exp(-((({}) * 2.5)**2.0)) - 1.0", x),
            GeometricActivationFunction::BipolarSigmoid => format!("2.0 / (1.0 + exp(-4.9 * ({}))) - 1.0", x),
            GeometricActivationFunction::Sine => format!("sin({})", x),
            GeometricActivationFunction::Cosine => format!("cos({})", x),
            GeometricActivationFunction::Constant1 => format!("1.0"),
        }
    }

    fn name(&self) -> String {
        match *self {
            GeometricActivationFunction::Linear => "Linear",
            GeometricActivationFunction::LinearBipolarClipped => "LinearBipolarClipped",
            GeometricActivationFunction::Absolute => "Absolute",
            GeometricActivationFunction::Gaussian => "Gaussian",
            GeometricActivationFunction::BipolarGaussian => "BipolarGaussian",
            GeometricActivationFunction::BipolarSigmoid =>  "BipolarSigmoid",
            GeometricActivationFunction::Sine => "Sine",
            GeometricActivationFunction::Cosine => "Consine",
            GeometricActivationFunction::Constant1 => "1.0",
        }.to_string()
    }
}


#[test]
fn test_bipolar_linear_clipped() {
    assert_eq!(0.0, GeometricActivationFunction::LinearBipolarClipped.calculate(0.0));
    assert_eq!(1.0, GeometricActivationFunction::LinearBipolarClipped.calculate(1.0));
    assert_eq!(-1.0, GeometricActivationFunction::LinearBipolarClipped.calculate(-1.0));
    assert_eq!(0.5, GeometricActivationFunction::LinearBipolarClipped.calculate(0.5));
    assert_eq!(-0.5, GeometricActivationFunction::LinearBipolarClipped.calculate(-0.5));
    assert_eq!(1.0, GeometricActivationFunction::LinearBipolarClipped.calculate(1.1));
    assert_eq!(-1.0, GeometricActivationFunction::LinearBipolarClipped.calculate(-1.1));
}

#[test]
fn test_constant1() {
    assert_eq!(1.0, GeometricActivationFunction::Constant1.calculate(0.0));
    assert_eq!(1.0, GeometricActivationFunction::Constant1.calculate(-1.0));
    assert_eq!(1.0, GeometricActivationFunction::Constant1.calculate(1.0));
}
