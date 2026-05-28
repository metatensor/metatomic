use json::JsonValue;

use crate::Error;

/// Options for the calculation of a pair list (neighbor list)
#[derive(Debug, Clone)]
pub struct PairListOptions {
    /// Cutoff radius for this pair list in the length unit of the model
    cutoff: f64,
    /// Whether the list is a full list (contains both the pair `i -> j` and `j -> i`)
    /// or a half list (contains only `i -> j`)
    full_list: bool,
    /// Whether the list guarantees that only atoms within the cutoff are
    /// included (strict) or may also include pairs slightly beyond the cutoff
    /// (non-strict)
    strict: bool,
    /// List of strings describing who requested this pair list
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

impl From<PairListOptions> for JsonValue {
    fn from(value: PairListOptions) -> Self {
        let mut result = JsonValue::new_object();
        result["type"] = "metatomic_pair_options".into();
        // store the bit pattern so the float round-trips exactly
        result["cutoff"] = format!("{:#x}", value.cutoff.to_bits()).into();
        result["full_list"] = value.full_list.into();
        result["strict"] = value.strict.into();
        result["requestors"] = value.requestors.into();
        return result;
    }
}

impl TryFrom<JsonValue> for PairListOptions {
    type Error = Error;

    fn try_from(value: JsonValue) -> Result<Self, Self::Error> {
        if !value.is_object() {
            return Err(Error::Serialization(
                "invalid JSON data for PairListOptions, expected an object".into()
            ));
        }

        if value["type"].as_str() != Some("metatomic_pair_options") {
            return Err(Error::Serialization(
                "'type' in JSON for PairListOptions must be 'metatomic_pair_options'".into()
            ));
        }

        let cutoff = value["cutoff"].as_str().ok_or_else(|| Error::Serialization(
            "'cutoff' in JSON for PairListOptions must be a hex-encoded string".into()
        ))?;
        let bits = u64::from_str_radix(cutoff.strip_prefix("0x").unwrap_or(cutoff), 16)
            .map_err(|_| Error::Serialization(
                "'cutoff' in JSON for PairListOptions must be a hex-encoded string".into()
            ))?;
        let cutoff = f64::from_bits(bits);

        if !cutoff.is_finite() || cutoff <= 0.0 {
            return Err(Error::Serialization(
                "'cutoff' in JSON for PairListOptions must be a finite positive number".into()
            ));
        }

        let full_list = value["full_list"].as_bool().ok_or_else(|| Error::Serialization(
            "'full_list' in JSON for PairListOptions must be a boolean".into()
        ))?;

        let strict = value["strict"].as_bool().ok_or_else(|| Error::Serialization(
            "'strict' in JSON for PairListOptions must be a boolean".into()
        ))?;

        let mut requestors = Vec::new();
        if value.has_key("requestors") {
            if !value["requestors"].is_array() {
                return Err(Error::Serialization(
                    "'requestors' in JSON for PairListOptions must be an array".into()
                ));
            }

            for requestor in value["requestors"].members() {
                let requestor = requestor.as_str().ok_or_else(|| Error::Serialization(
                    "'requestors' in JSON for PairListOptions must be an array of strings".into()
                ))?;
                // ignore empty strings and duplicates, keeping first-seen order
                if !requestor.is_empty() && !requestors.iter().any(|r| r == requestor) {
                    requestors.push(requestor.to_string());
                }
            }
        }

        return Ok(PairListOptions { cutoff, full_list, strict, requestors });
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


#[cfg(test)]
mod tests {
    mod pair_list_options {
        use super::super::*;

        fn example() -> PairListOptions {
            PairListOptions {
                cutoff: 3.5,
                full_list: true,
                strict: false,
                requestors: vec!["nl-1".to_string(), "nl-2".to_string()],
            }
        }

        #[test]
        fn roundtrip() {
            let options = example();
            let json: JsonValue = options.clone().into();

            assert_eq!(json["type"].as_str(), Some("metatomic_pair_options"));
            assert_eq!(json["cutoff"].as_str(), Some(format!("{:#x}", 3.5_f64.to_bits()).as_str()));
            assert_eq!(json["full_list"].as_bool(), Some(true));
            assert_eq!(json["strict"].as_bool(), Some(false));

            let parsed = PairListOptions::try_from(json).unwrap();
            assert_eq!(parsed.cutoff.to_bits(), options.cutoff.to_bits());
            assert_eq!(parsed.full_list, options.full_list);
            assert_eq!(parsed.strict, options.strict);
            assert_eq!(parsed.requestors, options.requestors);
        }

        #[test]
        fn cutoff_keeps_full_precision() {
            let mut options = example();
            options.cutoff = 1.0 / 3.0;
            let parsed = PairListOptions::try_from(JsonValue::from(options.clone())).unwrap();
            assert_eq!(parsed.cutoff.to_bits(), options.cutoff.to_bits());
        }

        #[test]
        fn requestors_are_optional() {
            let mut json: JsonValue = example().into();
            json.remove("requestors");
            let parsed = PairListOptions::try_from(json).unwrap();
            assert!(parsed.requestors.is_empty());
        }

        #[test]
        fn rejects_invalid_json() {
            // each case corrupts exactly one field of an otherwise valid object
            let with_cutoff = |value: f64| {
                let mut json = JsonValue::from(example());
                json["cutoff"] = format!("{:#x}", value.to_bits()).into();
                json
            };

            let mut wrong_type = JsonValue::from(example());
            wrong_type["type"] = "something-else".into();

            let mut missing_cutoff = JsonValue::from(example());
            missing_cutoff.remove("cutoff");

            let mut non_hex_cutoff = JsonValue::from(example());
            non_hex_cutoff["cutoff"] = "not-hex".into();

            let mut non_boolean_flag = JsonValue::from(example());
            non_boolean_flag["full_list"] = "yes".into();

            let mut non_array_requestors = JsonValue::from(example());
            non_array_requestors["requestors"] = "nl-1".into();

            let mut non_string_requestor = JsonValue::from(example());
            non_string_requestor["requestors"] = json::array![ "nl-1", 42 ];

            let cases = [
                (JsonValue::from("not an object"),
                    "invalid JSON data for PairListOptions, expected an object"),
                (wrong_type,
                    "'type' in JSON for PairListOptions must be 'metatomic_pair_options'"),
                (missing_cutoff,
                    "'cutoff' in JSON for PairListOptions must be a hex-encoded string"),
                (non_hex_cutoff,
                    "'cutoff' in JSON for PairListOptions must be a hex-encoded string"),
                (with_cutoff(f64::NAN),
                    "'cutoff' in JSON for PairListOptions must be a finite positive number"),
                (with_cutoff(f64::INFINITY),
                    "'cutoff' in JSON for PairListOptions must be a finite positive number"),
                (with_cutoff(-1.0),
                    "'cutoff' in JSON for PairListOptions must be a finite positive number"),
                (with_cutoff(0.0),
                    "'cutoff' in JSON for PairListOptions must be a finite positive number"),
                (non_boolean_flag,
                    "'full_list' in JSON for PairListOptions must be a boolean"),
                (non_array_requestors,
                    "'requestors' in JSON for PairListOptions must be an array"),
                (non_string_requestor,
                    "'requestors' in JSON for PairListOptions must be an array of strings"),
            ];

            for (json, expected) in cases {
                let error = PairListOptions::try_from(json).expect_err("expected an error");
                assert_eq!(error.to_string(), expected);
            }
        }

        #[test]
        fn requestors_skip_empty_and_duplicates() {
            let mut json: JsonValue = example().into();
            json["requestors"] = json::array![ "a", "", "b", "a" ];

            let parsed = PairListOptions::try_from(json).unwrap();
            assert_eq!(parsed.requestors, vec!["a".to_string(), "b".to_string()]);
        }
    }
}
