use std::path::PathBuf;

mod utils;

#[test]
fn run_cxx_tests() {
    const CARGO_TARGET_TMPDIR: &str = env!("CARGO_TARGET_TMPDIR");

    let mut build_dir = PathBuf::from(CARGO_TARGET_TMPDIR);
    build_dir.push("cxx-tests");
    std::fs::create_dir_all(&build_dir).expect("failed to create build dir");

    // ====================================================================== //
    // setup dependencies for the tests
    let deps_dir = build_dir.join("deps");
    let virtualenv_dir = deps_dir.join("virtualenv");
    std::fs::create_dir_all(&virtualenv_dir).expect("failed to create virtualenv dir");
    let python_exe = utils::create_python_venv(virtualenv_dir);
    let metatensor_cmake_prefix = utils::setup_metatensor_pip(&python_exe);

    // ====================================================================== //
    // build the metatomic C++ tests

    let cargo_manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let source_dir = cargo_manifest_dir.join("tests");

    // configure cmake for the tests
    let mut cmake_config = utils::cmake_config(&source_dir, &build_dir);
    cmake_config.arg("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON");
    cmake_config.arg(format!("-DCMAKE_PREFIX_PATH={}", metatensor_cmake_prefix.display()));
    utils::run_command(cmake_config, "cmake configuration");

    // build the tests
    let cmake_build = utils::cmake_build(&build_dir);
    utils::run_command(cmake_build, "cmake build");

    // ====================================================================== //
    // install the metatomic Python package, which also contains the Python
    // plugin, to test loading Python models from C++.

    let metatomic_prefix = deps_dir.join("usr");
    let cmake_install = utils::cmake_install(&build_dir, &metatomic_prefix);
    utils::run_command(cmake_install, "cmake install");

    let python_package_dir = cargo_manifest_dir.parent().unwrap().join("python").join("metatomic_core");
    let python_build_base = deps_dir.join("python-build");
    utils::setup_metatomic_core_pip_external(
        &python_exe, &python_package_dir, &metatomic_prefix, &python_build_base
    );

    let mut cmd = std::process::Command::new(&python_exe);
    cmd.arg("-c");
    cmd.arg("import metatomic; print(metatomic.utils.python_plugin_path)");
    let output = utils::run_command(cmd, "python to get the python plugin path");
    let python_plugin_path = String::from_utf8_lossy(&output.stdout).trim().to_string();

    // ====================================================================== //
    // run the tests

    let mut ctest = utils::ctest(&build_dir);
    ctest.env("METATOMIC_TESTS_PYTHON_PLUGIN", python_plugin_path);
    // make sure the Python plugin uses the virtualenv we just set up
    ctest.env("METATOMIC_PYTHON", &python_exe);
    utils::run_command(ctest, "ctest");
}
