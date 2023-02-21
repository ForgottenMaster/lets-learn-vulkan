use {
    naga::{
        back::{spv, spv::Writer},
        front::glsl::{Options, Parser},
        valid::{Capabilities, ValidationFlags, Validator},
        ShaderStage,
    },
    serde_derive::Serialize,
    std::{env, fs, path::PathBuf},
    tinytemplate::TinyTemplate,
};

#[derive(Serialize)]
struct Context {
    shaders: Vec<(String, String)>,
    open_brace: char,
    close_brace: char,
}

fn main() {
    // track paths for a recursive depth first search.
    let prefix: PathBuf = "assets/shaders".into();
    let mut paths: Vec<_> = vec![prefix.clone()];

    // parser for naga to use.
    let mut parser = Parser::default();

    // the names and codes for the shaders we process.
    let mut shaders = vec![];

    // while we've got more paths to recurse into.
    while let Some(path) = paths.pop() {
        if let Ok(files) = fs::read_dir(path) {
            for file in files {
                let file = file.unwrap();
                let file_type = file.file_type().unwrap();
                let file_path = file.path();
                if file_type.is_dir() {
                    paths.push(file_path);
                } else if file_type.is_file() {
                    if let Some(source) = read_and_compile_shader(file_path.clone(), &mut parser) {
                        let shader_name = file_path
                            .strip_prefix(prefix.clone())
                            .unwrap()
                            .to_str()
                            .unwrap()
                            .replace('\\', "/");
                        let source = format!("{source:?}");
                        shaders.push((shader_name, source));
                    }
                }
            }
        }
    }

    // write code out to file.
    const TEMPLATE: &str = "
    fn get_compiled_shader_mapping() -> ::std::collections::HashMap<&'static str, &'static [u32]> {open_brace}     
        let mut hash_map = ::std::collections::HashMap::new();
        {{ for shader in shaders }}
        {open_brace}
            const ARR: &[u32] = &{shader.1};
            hash_map.insert(\"{shader.0}\", ARR);
        {close_brace}
        {{ endfor }}
        hash_map
    {close_brace}
    ";
    let mut tt = TinyTemplate::new();
    tt.add_template("shaders", TEMPLATE).unwrap();
    let context = Context {
        shaders,
        open_brace: '{',
        close_brace: '}',
    };
    let code = tt.render("shaders", &context).unwrap();
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
