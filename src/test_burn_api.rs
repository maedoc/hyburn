use burn::backend::ndarray::NdArray;
use burn::tensor::Tensor;

#[test]
fn test_burn_exp_div() {
    type B = NdArray<f32>;
    let t = Tensor::<B, 1>::from_floats([0.0_f32, 1.0, 2.0], &Default::default());
    
    // exp
    let e = t.clone().exp();
    println!("exp: {:?}", e.into_data().as_slice::<f32>().unwrap());
    
    // division: tensor / scalar
    let half = t.clone() / 2.0;
    println!("div scalar: {:?}", half.into_data().as_slice::<f32>().unwrap());
    
    // division: scalar / tensor
    let one = Tensor::<B, 1>::ones([3], &Default::default());
    let recip = one / t.clone().exp();
    println!("recip: {:?}", recip.into_data().as_slice::<f32>().unwrap());
    
    // clamp
    let mut c = Tensor::<B, 1>::from_floats([-1.0_f32, 0.5, 2.0], &Default::default());
    c = c.clamp(0.0, 1.0);
    println!("clamp: {:?}", c.into_data().as_slice::<f32>().unwrap());
}
