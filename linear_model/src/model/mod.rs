pub mod dense;
pub mod model;
use crate::tensor::Tensor;

use ndarray::prelude::*;

pub trait Compute {
    fn compute_single(&self, x: Tensor) -> Tensor;
}
