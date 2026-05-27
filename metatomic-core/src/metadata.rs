use json::JsonValue;

use crate::Error;

/// TODO
pub struct PairListOptions {
    /// TODO
    cutoff: f64,
    /// TODO
    full_list: bool,
    /// TODO
    strict: bool,
    /// TODO
    requestors: Vec<String>,
}

impl std::cmp::PartialEq for PairListOptions {
    fn eq(&self, other: &Self) -> bool {
        self.cutoff == other.cutoff
            && self.full_list == other.full_list
            && self.strict == other.strict
    }
}

impl std::cmp::Eq for PairListOptions {}

impl std::cmp::PartialOrd for PairListOptions {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::cmp::Ord for PairListOptions {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cutoff.partial_cmp(&other.cutoff).expect("cutoff is NaN")
            .then_with(|| self.full_list.cmp(&other.full_list))
            .then_with(|| self.strict.cmp(&other.strict))
    }
}

// TODO
// {
//     "type": "metatomic_pair_options",
//     "cutoff": "0xaeabf23", <== hex of the int corresponding to the f64 bits to keep full precision
//     "full_list": false,
//     "strict": false,
//     "requestors": ["..."]
// }
impl From<PairListOptions> for JsonValue {
    fn from(value: PairListOptions) -> Self {
        todo!()
    }
}

impl TryFrom<JsonValue> for PairListOptions {
    type Error = Error;

    fn try_from(value: JsonValue) -> Result<Self, Self::Error> {
        todo!()
    }
}

// ========================================================================== //
// ========================================================================== //
// ========================================================================== //

/// TODO
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelMetadata {
    pub name: String,
    // TODO
}

// {
//     "type": "metatomic_model_metadata",
//     "name": "...",
//     "authors": ["..."],
//     "references": {
//         "implementation": ["..."],
//         "architecture": ["..."],
//         "model": ["..."]
//     },
//     "extra": {
//         "key...": "value..."
//     }
// },
impl From<ModelMetadata> for JsonValue {
    fn from(value: ModelMetadata) -> Self {
        todo!()
    }
}

impl TryFrom<JsonValue> for ModelMetadata {
    type Error = Error;

    fn try_from(value: JsonValue) -> Result<Self, Self::Error> {
        todo!()
    }
}

// ========================================================================== //
// ========================================================================== //
// ========================================================================== //

/// TODO, previously `ModelOutput`
#[derive(Debug)]
pub struct Quantity {
    pub name: String,
    // TODO
}

// TODO:
// {
//     "type": "metatomic_quantity",
//     "name": "...",
//     "unit": "...",
//     "gradients": ["...", "..."],
//     "sample_kind": "atom" | "system" | "atom-pair",
// },
impl From<Quantity> for JsonValue {
    fn from(value: Quantity) -> Self {
        todo!()
    }
}

impl TryFrom<JsonValue> for Quantity {
    type Error = Error;

    fn try_from(value: JsonValue) -> Result<Self, Self::Error> {
        todo!()
    }
}
