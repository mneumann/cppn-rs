/// Represents a position within a substrate.
/// Each position coordinate is mapped to an input of a CPPN.
pub trait Position {
    const DIMENSIONS: usize;
    fn coords(&self) -> &[f64];
    fn distance_square(&self, other: &Self) -> f64;
    fn distance(&self, other: &Self) -> f64 {
        self.distance_square(other).sqrt()
    }
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
    pub fn new(x: f64, y: f64) -> Self {
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
    const DIMENSIONS: usize = 2;
    #[inline(always)]
    fn coords(&self) -> &[f64] {
        &self.0
    }

    #[inline(always)]
    fn origin() -> Self {
        Position2d::new(0.0, 0.0)
    }

    #[inline]
    fn distance_square(&self, other: &Self) -> f64 {
        (self.x() - other.x()).powi(2) + (self.y() - other.y()).powi(2)
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


pub struct Position3d([f64; 3]);

impl Position3d {
    #[inline(always)]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Position3d([x, y, z])
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
    pub fn z(&self) -> f64 {
        self.0[2]
    }

    #[inline(always)]
    pub fn xyz(&self) -> (f64, f64, f64) {
        (self.0[0], self.0[1], self.0[2])
    }
}

impl Position for Position3d {
    const DIMENSIONS: usize = 3;

    #[inline(always)]
    fn coords(&self) -> &[f64] {
        &self.0
    }

    #[inline(always)]
    fn origin() -> Self {
        Position3d::new(0.0, 0.0, 0.0)
    }

    #[inline]
    fn distance_square(&self, other: &Self) -> f64 {
        (self.x() - other.x()).powi(2) + (self.y() - other.y()).powi(2) +
        (self.z() - other.z()).powi(2)
    }
}

#[test]
fn test_position3d_distance() {
    assert_eq!(0.0, Position3d::origin().distance(&Position3d::origin()));
    assert_eq!(1.0,
               Position3d::origin().distance(&Position3d::new(1.0, 0.0, 0.0)));
    assert_eq!(2.0,
               Position3d::origin().distance(&Position3d::new(2.0, 0.0, 0.0)));
    assert_eq!((2.0f64).sqrt(),
               Position3d::origin().distance(&Position3d::new(1.0, 0.0, 1.0)));
    assert_eq!((2.0f64).sqrt(),
               Position3d::origin().distance(&Position3d::new(1.0, 0.0, -1.0)));
    assert_eq!((2.0f64).sqrt(),
               Position3d::origin().distance(&Position3d::new(-1.0, 0.0, -1.0)));
}

#[test]
fn test_interpolate_one_axis() {
    let a = Position2d::new(-1.0, 0.0);
    let b = Position2d::new(1.0, 0.0);

    assert_eq!((-1.0, 0.0), a.interpolate(&b, 0.0).xy());
    assert_eq!((0.0, 0.0), a.interpolate(&b, 0.5).xy());
    assert_eq!((1.0, 0.0), a.interpolate(&b, 1.0).xy());
}

#[test]
fn test_interpolate_two_axes() {
    let a = Position2d::new(-1.0, 1.0);
    let b = Position2d::new(1.0, -1.0);

    assert_eq!((-1.0, 1.0), a.interpolate(&b, 0.0).xy());
    assert_eq!((0.0, 0.0), a.interpolate(&b, 0.5).xy());
    assert_eq!((1.0, -1.0), a.interpolate(&b, 1.0).xy());
}

#[test]
fn test_interpolate_multi() {
    let a = Position2d::new(-1.0, 1.0);
    let b = Position2d::new(1.0, -1.0);

    assert_eq!((-1.0, -1.0),
               a.interpolate_multi(&b, &Position2d::new(0.0, 1.0)).xy());
    assert_eq!((0.0, 0.0),
               a.interpolate_multi(&b, &Position2d::new(0.5, 0.5)).xy());
    assert_eq!((1.0, 1.0),
               a.interpolate_multi(&b, &Position2d::new(1.0, 0.0)).xy());
}
