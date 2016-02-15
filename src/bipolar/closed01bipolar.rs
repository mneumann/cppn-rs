use std::fmt::Debug;
use std::convert::Into;

/// Encapsulates a floating point number in the range [-1, 1] including both endpoints.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Closed01Bipolar<F>(F) where F: Copy + Clone + Debug + PartialEq + PartialOrd;

impl Into<f64> for Closed01Bipolar<f64> {
    fn into(self) -> f64 {
        self.get()
    }
}

impl Closed01Bipolar<f64> {
    #[inline(always)]
    pub fn new(f: f64) -> Closed01Bipolar<f64> {
        if f >= -1.0 && f <= 1.0 {
            Closed01Bipolar(f)
        } else {
            panic!("assertion failed: f >= -1.0 && f <= 1.0; f = {}", f);
        }
    }

    #[inline(always)]
    /// Creates a new Closed01Bipolar from `f` clipped to the range [-1, 1].
    pub fn new_clipped(f: f64) -> Closed01Bipolar<f64> {
        Closed01Bipolar::new_debug_checked(if f > 1.0 {
            1.0
        } else if f < -1.0 {
            -1.0
        } else {
            f
        })
    }

    #[inline(always)]
    fn new_debug_checked(f: f64) -> Closed01Bipolar<f64> {
        debug_assert!(f >= -1.0 && f <= 1.0);
        Closed01Bipolar(f)
    }

    #[inline(always)]
    pub fn zero() -> Closed01Bipolar<f64> {
        Closed01Bipolar::new_debug_checked(0.0)
    }

    #[inline(always)]
    pub fn negative_one() -> Closed01Bipolar<f64> {
        Closed01Bipolar::new_debug_checked(-1.0)
    }

    #[inline(always)]
    pub fn one() -> Closed01Bipolar<f64> {
        Closed01Bipolar::new_debug_checked(1.0)
    }

    #[inline(always)]
    /// Returns the smaller of the two.
    pub fn min(self, other: Closed01Bipolar<f64>) -> Closed01Bipolar<f64> {
        if self.0 <= other.0 {
            self
        } else {
            other
        }
    }

    #[inline(always)]
    /// Returns the greater of the two.
    pub fn max(self, other: Closed01Bipolar<f64>) -> Closed01Bipolar<f64> {
        if self.0 >= other.0 {
            self
        } else {
            other
        }
    }

    #[inline(always)]
    /// Returns the wrapped value.
    pub fn get(self) -> f64 {
        debug_assert!(self.0 >= -1.0 && self.0 <= 1.0);
        self.0
    }

    #[inline(always)]
    /// Returns the negated value
    pub fn negate(self) -> Closed01Bipolar<f64> {
        Closed01Bipolar::new_debug_checked(-self.0)
    }

    #[inline(always)]
    /// Multiplies both numbers
    pub fn mul(self, scalar: Closed01Bipolar<f64>) -> Closed01Bipolar<f64> {
        Closed01Bipolar::new_debug_checked(self.get() * scalar.get())
    }
}
