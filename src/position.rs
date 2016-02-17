/// Represents a position within a substrate.
/// Each position coordinate is mapped to an input of a CPPN.
pub trait Position {
    fn coords(&self) -> &[f64];
    fn distance(&self, other: &Self) -> f64;
}

pub struct Position2d([f64; 2]);

impl Position2d {
    #[inline(always)]
    pub fn new(x: f64, y: f64) -> Position2d {
        Position2d([x, y])
    }

    #[inline(always)]
    pub fn x(&self) -> f64 {
        self.0[0]
    }

    #[inline(always)]
    pub fn y(&self) -> f64 {
        self.0[1]
    }
}

impl Position for Position2d {
    #[inline(always)]
    fn coords(&self) -> &[f64] {
        &self.0
    }

    #[inline]
    fn distance(&self, other: &Self) -> f64 {
        ((self.x() - other.x()).powi(2) + (self.y() - other.y()).powi(2)).sqrt()
    }
}
