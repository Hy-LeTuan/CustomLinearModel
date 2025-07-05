pub mod dense;
pub mod model;
use ndarray::prelude::*;

pub trait Compute {
    fn compute_single(
        &self,
        x: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix2>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix2>;
}
