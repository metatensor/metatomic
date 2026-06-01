use json::JsonValue;

use crate::Error;

static STANDARD_QUANTITIES: &[&str] = &[
    "charge",
    "energy_ensemble",
    "energy_uncertainty",
    "energy",
    "feature",
    "heat_flux",
    "mass",
    "momentum",
    "non_conservative_force",
    "non_conservative_stress",
    "position",
    "spin_multiplicity",
    "velocity",
];

fn is_valid_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let first = s.chars().next().unwrap();
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Validate a quantity name.
///
/// The name can be either a standard name or a custom name with the form
/// `<namespace>::<name>`, where the namespace can itself contain `::` to define
/// sub-namespaces.
///
/// Both standard and custom names can also define a variant with the form
/// `<name>/<variant>` or `<namespace>::<name>/<variant>`.
///
/// All components (namespace, name, variant) must be non-empty if they are
/// present, and must be valid identifiers (alphanumeric + underscore, not
/// starting with a digit).
pub(crate) fn validate_quantity_name(name: &str) -> Result<(), Error> {
    if STANDARD_QUANTITIES.contains(&name) {
        return Ok(());
    }

    let (main_part, variant) = if let Some(pos) = name.find('/') {
        (&name[..pos], Some(&name[pos + 1..]))
    } else {
        (name, None)
    };

    if main_part.is_empty() {
        return Err(Error::InvalidParameter(format!(
            "quantity name cannot be empty in '{}'", name
        )));
    }

    if let Some(variant) = variant {
        if !is_valid_identifier(variant) {
            return Err(Error::InvalidParameter(format!(
                "invalid quantity variant '{}' in '{}': must be a valid identifier (alphanumeric or underscore, not starting with a digit)",
                variant, name
            )));
        }
    }

    if STANDARD_QUANTITIES.contains(&main_part) {
        return Ok(());
    }

    let components: Vec<_> = main_part.split("::").collect();
    for component in &components {
        if !is_valid_identifier(component) {
            return Err(Error::InvalidParameter(format!(
                "invalid quantity name component '{}' in '{}': must be a valid identifier (alphanumeric or underscore, not starting with a digit)",
                component, name
            )));
        }
    }

    if components.len() == 1 {
        return Err(Error::InvalidParameter(format!(
            "'{}' is not a standard quantity name; custom quantity names must use '<namespace>::<name>'",
            name
        )));
    }

    Ok(())
}


/// Different kind of samples a quantity can be associated with
#[derive(Debug, Clone, PartialEq)]
pub enum SampleKind {
    /// The quantity is defined for each atom (e.g. atomic energy, charge, ...)
    Atom,
    /// The quantity is defined for the whole system (e.g. total energy, ...)
    System,
    /// The quantity is defined for each pair of atoms (e.g. hamiltonian elements, ...)
    AtomPair,
}

impl From<SampleKind> for JsonValue {
    fn from(value: SampleKind) -> Self {
        let s = match value {
            SampleKind::Atom => "atom",
            SampleKind::System => "system",
            SampleKind::AtomPair => "atom_pair",
        };
        JsonValue::from(s)
    }
}

impl<'a> TryFrom<&'a JsonValue> for SampleKind {
    type Error = Error;

    fn try_from(value: &'a JsonValue) -> Result<Self, Self::Error> {
        let s = value.as_str().ok_or_else(|| Error::Serialization(
            "'sample_kind' in JSON for Quantity must be a string".into()
        ))?;
        match s {
            "atom" => Ok(SampleKind::Atom),
            "system" => Ok(SampleKind::System),
            "atom_pair" => Ok(SampleKind::AtomPair),
            _ => Err(Error::Serialization(format!(
                "'sample_kind' in JSON for Quantity must be 'atom', 'system' or 'atom_pair', got '{}'", s
            ))),
        }
    }
}

/// Different gradients that a quantity can have
#[derive(Debug, Clone, PartialEq)]
pub enum Gradients {
    /// Gradients with respect to atomic positions
    Positions,
    /// Gradients with respect to the strain (typically used for stress)
    Strain,
}

impl From<Gradients> for JsonValue {
    fn from(value: Gradients) -> Self {
        let s = match value {
            Gradients::Positions => "positions",
            Gradients::Strain => "strain",
        };
        JsonValue::from(s)
    }
}

impl<'a> TryFrom<&'a JsonValue> for Gradients {
    type Error = Error;

    fn try_from(value: &'a JsonValue) -> Result<Self, Self::Error> {
        let s = value.as_str().ok_or_else(|| Error::Serialization(
            "'gradients' in JSON for Quantity must be a string".into()
        ))?;
        match s {
            "positions" => Ok(Gradients::Positions),
            "strain" => Ok(Gradients::Strain),
            _ => Err(Error::Serialization(format!(
                "'gradients' in JSON for Quantity must be 'positions' or 'strain', got '{}'", s
            ))),
        }
    }
}

/// A quantity that a model can use as input or output
#[derive(Debug, Clone)]
pub struct Quantity {
    /// Name of the quantity, this can be a standard name from
    /// <https://docs.metatensor.org/metatomic/latest/quantities/index.html>, or
    /// a custom name of the form `<namespace>::<name>[/<variant>]`
    pub name: String,
    /// Unit of the quantity
    pub unit: String,
    /// Description of the quantity, used to provide more details about the
    /// quantity, especially when a model defines multiple variants of the same
    /// quantity.
    pub description: Option<String>,
    /// List of explicit gradients for this quantity, stored in the
    /// corresponding `TensorMap`
    pub gradients: Vec<Gradients>,
    /// The kind of samples this quantity is associated with (e.g. per-atom,
    /// per-system, ...)
    pub sample_kind: SampleKind,
}

impl From<Quantity> for JsonValue {
    fn from(value: Quantity) -> Self {
        let mut result = JsonValue::new_object();
        result["type"] = "metatomic_quantity".into();
        result["name"] = value.name.into();
        result["unit"] = value.unit.into();
        if let Some(description) = value.description {
            result["description"] = description.into();
        }
        result["gradients"] = value.gradients.into();
        result["sample_kind"] = value.sample_kind.into();
        return result;
    }
}


impl<'a> TryFrom<&'a JsonValue> for Quantity {
    type Error = Error;

    fn try_from(value: &'a JsonValue) -> Result<Self, Self::Error> {
        if !value.is_object() {
            return Err(Error::Serialization(
                "invalid JSON data for Quantity, expected an object".into()
            ));
        }

        if value["type"].as_str() != Some("metatomic_quantity") {
            return Err(Error::Serialization(
                "'type' in JSON for Quantity must be 'metatomic_quantity'".into()
            ));
        }

        let name = value["name"].as_str().ok_or_else(|| Error::Serialization(
            "'name' in JSON for Quantity must be a string".into()
        ))?;
        validate_quantity_name(name)?;

        let unit = value["unit"].as_str().ok_or_else(|| Error::Serialization(
            "'unit' in JSON for Quantity must be a string".into()
        ))?;

        let mut description = value["description"].as_str().map(|s| s.to_string());
        if description == Some(String::new()) {
            // Treat empty description as None
            description = None;
        }

        let gradients = &value["gradients"];
        if !gradients.is_array() {
            return Err(Error::Serialization(
                "'gradients' in JSON for Quantity must be an array".into()
            ));
        }
        let gradients = gradients.members()
            .map(Gradients::try_from)
            .collect::<Result<Vec<_>, _>>()?;

        let sample_kind = SampleKind::try_from(&value["sample_kind"])?;

        Ok(Quantity {
            name: name.to_string(),
            unit: unit.to_string(),
            description,
            gradients,
            sample_kind,
        })
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn example() -> Quantity {
        Quantity {
            name: "energy".into(),
            unit: "eV".into(),
            description: Some("total energy of the system".into()),
            gradients: vec![Gradients::Positions],
            sample_kind: SampleKind::Atom,
        }
    }

    #[test]
    fn roundtrip() {
        let quantity = example();
        let json: JsonValue = quantity.into();

        assert_eq!(json["type"].as_str(), Some("metatomic_quantity"));
        assert_eq!(json["name"].as_str(), Some("energy"));
        assert_eq!(json["unit"].as_str(), Some("eV"));
        assert_eq!(json["gradients"][0].as_str(), Some("positions"));
        assert_eq!(json["sample_kind"].as_str(), Some("atom"));

        let parsed = Quantity::try_from(&json).unwrap();
        assert_eq!(parsed.name, "energy");
        assert_eq!(parsed.unit, "eV");
        assert_eq!(parsed.gradients, vec![Gradients::Positions]);
        assert!(matches!(parsed.sample_kind, SampleKind::Atom));
    }

    #[test]
    fn roundtrip_all_variants() {
        for sample in [SampleKind::Atom, SampleKind::System, SampleKind::AtomPair] {
            for grads in [
                vec![],
                vec![Gradients::Positions],
                vec![Gradients::Strain],
                vec![Gradients::Positions, Gradients::Strain],
            ] {
                let quantity = Quantity {
                    name: "test_ns::test".into(),
                    unit: "unit".into(),
                    description: Some("Hello".to_string()),
                    gradients: grads.clone(),
                    sample_kind: sample.clone(),
                };
                let parsed = Quantity::try_from(&JsonValue::from(quantity.clone())).unwrap();
                assert_eq!(parsed.name, quantity.name);
                assert_eq!(parsed.unit, quantity.unit);
                assert_eq!(parsed.gradients, grads);
                assert_eq!(parsed.sample_kind, sample);
            }
        }
    }

    #[test]
    fn rejects_invalid_json() {
        let mut wrong_type = JsonValue::from(example());
        wrong_type["type"] = "something-else".into();

        let mut missing_name = JsonValue::from(example());
        missing_name.remove("name");

        let mut missing_unit = JsonValue::from(example());
        missing_unit.remove("unit");

        let mut missing_gradients = JsonValue::from(example());
        missing_gradients.remove("gradients");

        let mut non_array_gradients = JsonValue::from(example());
        non_array_gradients["gradients"] = "positions".into();

        let mut invalid_gradient = JsonValue::from(example());
        invalid_gradient["gradients"] = json::array!["positions", "foo"];

        let mut missing_sample_kind = JsonValue::from(example());
        missing_sample_kind.remove("sample_kind");

        let mut invalid_sample_kind = JsonValue::from(example());
        invalid_sample_kind["sample_kind"] = "foo".into();

        let cases: Vec<(JsonValue, &str)> = vec![
            (JsonValue::from("not an object"),
                "serialization error: invalid JSON data for Quantity, expected an object"),
            (wrong_type,
                "serialization error: 'type' in JSON for Quantity must be 'metatomic_quantity'"),
            (missing_name,
                "serialization error: 'name' in JSON for Quantity must be a string"),
            (missing_unit,
                "serialization error: 'unit' in JSON for Quantity must be a string"),
            (missing_gradients,
                "serialization error: 'gradients' in JSON for Quantity must be an array"),
            (non_array_gradients,
                "serialization error: 'gradients' in JSON for Quantity must be an array"),
            (invalid_gradient,
                "serialization error: 'gradients' in JSON for Quantity must be 'positions' or 'strain', got 'foo'"),
            (missing_sample_kind,
                "serialization error: 'sample_kind' in JSON for Quantity must be a string"),
            (invalid_sample_kind,
                "serialization error: 'sample_kind' in JSON for Quantity must be 'atom', 'system' or 'atom_pair', got 'foo'"),
        ];

        for (json, expected) in cases {
            let error = Quantity::try_from(&json).expect_err("expected an error");
            assert_eq!(error.to_string(), expected);
        }
    }

    #[test]
    fn validate_names() {
        for name in STANDARD_QUANTITIES {
            assert!(validate_quantity_name(name).is_ok(), "expected '{}' to be valid", name);
        }

        let custom = [
            "my_model::energy",
            "org::my_model::custom_qty",
            "ns1::ns2::ns3::energy",
            "some_ns::name_with_underscores",
            "_ns::_name",
        ];
        for name in custom {
            assert!(validate_quantity_name(name).is_ok(), "expected '{}' to be valid", name);
        }

        let variants = [
            "energy/ensemble",
            "my_ns::energy/raw",
            "ns1::ns2::energy/some_variant",
        ];
        for name in variants {
            assert!(validate_quantity_name(name).is_ok(), "expected '{}' to be valid", name);
        }

        let error = validate_quantity_name("").expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: quantity name cannot be empty in ''");

        let error = validate_quantity_name("not_a_standard_name").expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: 'not_a_standard_name' is not a standard quantity name; custom quantity names must use '<namespace>::<name>'");

        let error = validate_quantity_name("/variant").expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: quantity name cannot be empty in '/variant'");

        let error = validate_quantity_name("name/").expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: invalid quantity variant '' in 'name/': must be a valid identifier (alphanumeric or underscore, not starting with a digit)");

        let error = validate_quantity_name("::energy").expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: invalid quantity name component '' in '::energy': must be a valid identifier (alphanumeric or underscore, not starting with a digit)");

        let error = validate_quantity_name("ns::").expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: invalid quantity name component '' in 'ns::': must be a valid identifier (alphanumeric or underscore, not starting with a digit)");

        let error = validate_quantity_name("ns::/variant").expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: invalid quantity name component '' in 'ns::/variant': must be a valid identifier (alphanumeric or underscore, not starting with a digit)");

        let error = validate_quantity_name("::").expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: invalid quantity name component '' in '::': must be a valid identifier (alphanumeric or underscore, not starting with a digit)");

        let error = validate_quantity_name("123name").expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: invalid quantity name component '123name' in '123name': must be a valid identifier (alphanumeric or underscore, not starting with a digit)");

        let error = validate_quantity_name("my_ns::123name").expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: invalid quantity name component '123name' in 'my_ns::123name': must be a valid identifier (alphanumeric or underscore, not starting with a digit)");

        let error = validate_quantity_name("my_ns::name/123variant").expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: invalid quantity variant '123variant' in 'my_ns::name/123variant': must be a valid identifier (alphanumeric or underscore, not starting with a digit)");

        let error = validate_quantity_name("has spaces").expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: invalid quantity name component 'has spaces' in 'has spaces': must be a valid identifier (alphanumeric or underscore, not starting with a digit)");

        let error = validate_quantity_name("my_ns::name/has spaces").expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: invalid quantity variant 'has spaces' in 'my_ns::name/has spaces': must be a valid identifier (alphanumeric or underscore, not starting with a digit)");

        let error = validate_quantity_name("has-dash").expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: invalid quantity name component 'has-dash' in 'has-dash': must be a valid identifier (alphanumeric or underscore, not starting with a digit)");
    }
}
