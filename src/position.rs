/// Represents a position within a substrate.
/// Each position coordinate is mapped to an input of a CPPN.
pub trait Position {
    fn coords(&self) -> &[f64];
    fn distance(&self, other: &Self) -> f64;
    fn origin() -> Self;
}

pub trait Interpolate {
    /// Linearly interpolates between `self` and `other`.
    /// For t = 0.0 this should return `self`. For t = 1.0 this
    /// should return `other`.
    fn interpolate(&self, other: &Self, t: f64) -> Self;

    fn interpolate_multi(&self, other: &Self, t: &Self) -> Self;
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

    #[inline(always)]
    pub fn xy(&self) -> (f64, f64) {
        (self.0[0], self.0[1])
    }

}

impl Position for Position2d {
    #[inline(always)]
    fn coords(&self) -> &[f64] {
        &self.0
    }

    #[inline(always)]
    fn origin() -> Self {
        Position2d::new(0.0, 0.0)
    }

    #[inline]
    fn distance(&self, other: &Self) -> f64 {
        ((self.x() - other.x()).powi(2) + (self.y() - other.y()).powi(2)).sqrt()
    }
}

impl Interpolate for Position2d {
    fn interpolate(&self, other: &Self, t: f64) -> Self {
        let x = self.x() * (1.0 - t) + other.x() * t;
        let y = self.y() * (1.0 - t) + other.y() * t;
        Position2d([x, y])
    }

    fn interpolate_multi(&self, other: &Self, t: &Self) -> Self {
        let tx = t.x();
        let ty = t.y();
        let x = self.x() * (1.0 - tx) + other.x() * tx;
        let y = self.y() * (1.0 - ty) + other.y() * ty;
        Position2d([x, y])
    }
}

#[test]
fn test_interpolate_one_axis() {
    let a = Position2d::new(-1.0, 0.0);
    let b = Position2d::new(1.0, 0.0);

    assert_eq!((-1.0, 0.0), a.interpolate(&b, 0.0).xy());
    assert_eq!((0.0, 0.0),  a.interpolate(&b, 0.5).xy());
    assert_eq!((1.0, 0.0),  a.interpolate(&b, 1.0).xy());
}

#[test]
fn test_interpolate_two_axes() {
    let a = Position2d::new(-1.0, 1.0);
    let b = Position2d::new(1.0, -1.0);

    assert_eq!((-1.0, 1.0), a.interpolate(&b, 0.0).xy());
    assert_eq!((0.0, 0.0),  a.interpolate(&b, 0.5).xy());
    assert_eq!((1.0, -1.0),  a.interpolate(&b, 1.0).xy());
}

#[test]
fn test_interpolate_multi() {
    let a = Position2d::new(-1.0, 1.0);
    let b = Position2d::new(1.0, -1.0);

    assert_eq!((-1.0, -1.0), a.interpolate_multi(&b, &Position2d::new(0.0, 1.0)).xy());
    assert_eq!((0.0, 0.0),  a.interpolate_multi(&b, &Position2d::new(0.5, 0.5)).xy());
    assert_eq!((1.0, 1.0),  a.interpolate_multi(&b, &Position2d::new(1.0, 0.0)).xy());
}
