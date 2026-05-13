use std::path::PathBuf;
use std::sync::Mutex;

mod utils;

lazy_static::lazy_static! {
    // Make sure only one of the tests below run at the time, since they both
    // try to modify the same files
    static ref LOCK: Mutex<()> = Mutex::new(());
}


/// Check that metatomic can be built and installed with cmake, and that the
/// installed version can be used from another cmake project with `find_package`
#[test]
fn check_cxx_install() {
    let _guard = match LOCK.lock() {
        Ok(guard) => guard,
        Err(_) => {
            panic!("another test failed, stopping")
        }
    };

    const CARGO_TARGET_TMPDIR: &str = env!("CARGO_TARGET_TMPDIR");

    // ====================================================================== //
    // build and install metatensor with cmake
    let mut build_dir = PathBuf::from(CARGO_TARGET_TMPDIR);
    build_dir.push("cxx-install");
    build_dir.push("cmake-find-package");
    std::fs::create_dir_all(&build_dir).expect("failed to create build dir");

    let deps_dir = build_dir.join("deps");
    let virtualenv_dir = deps_dir.join("virtualenv");
    std::fs::create_dir_all(&virtualenv_dir).expect("failed to create virtualenv dir");
    let python_exe = utils::create_python_venv(virtualenv_dir);
    let metatensor_cmake_prefix = utils::setup_metatensor_pip(&python_exe);

    let metatomic_dep = deps_dir.join("metatomic-core");
    let source_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    let cmake_args = vec![
        format!("-DCMAKE_PREFIX_PATH={}", metatensor_cmake_prefix.display()),
    ];
    let metatomic_cmake_prefix = utils::setup_metatomic_cmake(&source_dir, &metatomic_dep, cmake_args);

    // ====================================================================== //
    // try to use the installed metatomic from cmake
    let mut tests_source_dir = source_dir;
    tests_source_dir.extend(["tests", "cmake-project"]);

    // configure cmake for the test cmake project
    let mut cmake_config = utils::cmake_config(&tests_source_dir, &build_dir);
    cmake_config.arg(format!("-DCMAKE_PREFIX_PATH={};{}", metatensor_cmake_prefix.display(), metatomic_cmake_prefix.display()));
    utils::run_command(cmake_config, "cmake configuration");

    // build the code, linking to metatensor
    let cmake_build = utils::cmake_build(&build_dir);
    utils::run_command(cmake_build, "cmake build");

    // run the executables
    let ctest = utils::ctest(&build_dir);
    utils::run_command(ctest, "ctest");
}
