pub mod dataset;
pub mod loss;
pub mod model;

use crate::model::Compute;
use model::model::Model;

use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

fn main() {
    let (_, _) = dataset::read_dataset();

    let num_elem = 10;
    let input_dim = 5;

    let model = Model::new(vec![input_dim, 16, 32, 64, 32, 16], 1);

    let a = Array::random((num_elem, input_dim), Uniform::new(0.0, 1.0));
    let target = Array::random((10, 1), Uniform::new(0.0, 1.0));

    let res = model.compute_single(a);

    let loss = loss::lse_loss(
        &res,
        &target,
        model.get_weight(6 - 1).expect("Invalid layer index"),
    );

    // println!("Model: {}", model);
    println!("Res: {:?} || Shape of res: {:?}", res, res.shape());
    println!("Loss: {:?}", loss);
}
