use ndarray::prelude::*;

pub fn lse_loss(
    input: &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Ix2>,
    target: &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Ix2>,
    weight: &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Ix2>,
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> {
    assert_eq!(
        input.shape()[input.shape().len() - 1],
        weight.shape()[weight.shape().len() - 1],
        "Mismatch shape in calculating loss. Input has shape: {:?} but weight has shape: {:?}",
        input.shape(),
        weight.shape()
    );
    let loss = target - input.dot(&weight.t());
    let loss = 0.5 * loss.pow2();

    return loss;
}

pub fn lse_loss_derivative(
    input: &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Ix2>,
    loss: &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Ix2>,
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Ix2> {
    let res = -1.0 * (input.dot(loss));
    return res;
}
