use {
    naga::{
        back::{spv, spv::Writer},
        front::glsl::{Options, Parser},
        valid::{Capabilities, ValidationFlags, Validator},
        ShaderStage,
    },
    std::{env, fs, path::PathBuf},
};

fn main() {
    // string for the Rust source code to write to access the compiled shader code from
    // hash map.
    let mut code = String::from(
        "
    fn get_compiled_shader_map() -> ::std::collections::HashMap<&'static str, Vec<u32>> {
        let mut map = ::std::collections::HashMap::new();
    ",
    );

    // track paths for a recursive depth first search.
    let prefix: PathBuf = "assets/shaders".into();
    let mut paths: Vec<_> = vec![prefix.clone()];

    // parser for naga
    let mut parser = Parser::default();

    // while we've got more paths to recurse into.
    while let Some(path) = paths.pop() {
        if let Ok(files) = fs::read_dir(path) {
            for file in files {
                let file = file.unwrap(); // should exist here because we're reading it directly from read_dir.
                let file_type = file.file_type().unwrap(); // every file will have a file type.
                let file_path = file.path();
                if file_type.is_dir() {
                    paths.push(file_path);
                } else if file_type.is_file() {
                    if let Some(source) = read_and_compile_shader(file_path.clone(), &mut parser) {
                        let file_path = file_path
                            .strip_prefix(prefix.clone())
                            .unwrap()
                            .to_str()
                            .unwrap()
                            .replace("\\", "/");
                        code.push_str(&format!(
                            "map.insert(\"{file_path}\", vec![{}]);",
                            words_to_string(source)
                        ));
                    }
                }
            }
        }
    }

    // close off the shader function.
    code.push_str("map}");

    // write code out to file.
    let path = format!("{}/shaders.rs", env::var("OUT_DIR").unwrap());
    fs::write(path, code).unwrap();

    // watch for changes to the shader sources before re-running.
    println!("cargo:rerun-if-changed=\"assets/shaders\"");
}

fn read_and_compile_shader(input_path: PathBuf, parser: &mut Parser) -> Option<Vec<u32>> {
    let extension = input_path.extension()?.to_str()?;
    let source = fs::read_to_string(&input_path).ok()?;
    let options = match extension {
        "frag" => Options::from(ShaderStage::Fragment),
        "vert" => Options::from(ShaderStage::Vertex),
        _ => return None,
    };
    let module = parser.parse(&options, &source).ok()?;
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .ok()?;
    let mut words = vec![];
    let mut writer = Writer::new(&spv::Options::default()).ok()?;
    writer.write(&module, &info, None, &mut words).ok()?;
    Some(words)
}

fn words_to_string(words: Vec<u32>) -> String {
    let mut string = String::new();
    let words_len = words.len();
    for (idx, word) in words.into_iter().enumerate() {
        if idx < words_len - 1 {
            string.push_str(&format!("{word}, "));
        } else {
            string.push_str(&format!("{word}"));
        }
    }
    string
}
