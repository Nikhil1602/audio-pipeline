//! Prints the declared inputs and outputs of ONNX models (development aid).
//!
//! `cargo run -p sd-vad --example onnx_io -- model.onnx [more.onnx ...]`

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for path in std::env::args().skip(1) {
        let session =
            ort::session::Session::builder(sd_vad::runtime()?)?.commit_from_file(&path)?;
        println!("{path}");
        for i in session.inputs() {
            println!("  in  {}: {:?}", i.name(), i.dtype());
        }
        for o in session.outputs() {
            println!("  out {}: {:?}", o.name(), o.dtype());
        }
    }
    Ok(())
}
