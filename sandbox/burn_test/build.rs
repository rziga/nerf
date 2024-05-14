use burn_import::{
    onnx::ModelGen,
    burn::graph::RecordType
};

fn main() {
    // Generate Rust code from the ONNX model file
    ModelGen::new()
        .input("src/model/nelf.onnx")
        .out_dir("model/")
        .record_type(RecordType::Bincode)
        .embed_states(true)
        .run_from_script();
}