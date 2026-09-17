mod inputs;

#[allow(clippy::module_inception)]
mod model;
pub use self::model::Model;

mod execute;
pub use self::execute::execute_model;
