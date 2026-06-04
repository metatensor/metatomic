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

/// The name of a quantity, which can be either a standard name or a custom name
/// with an optional variant.
///
/// This struct enforces that the name is either a known standard name or a
/// custom name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuantityName {
    /// The full name of the quantity, including namespace and variant if present
    full: String,
    /// Optional namespace for custom quantity names. Standard quantity names do
    /// not have a namespace.
    namespace: Option<String>,
    /// The base name of the quantity
    base: String,
    /// Optional variant of the quantity, i.e. `pbe0` in `energy/pbe0`
    variant: Option<String>,
}

impl QuantityName {
    /// Parse and validate a quantity name.
    ///
    /// The name can be either a standard name or a custom name with the form
    /// `<namespace>::<name>`, where the namespace can itself contain `::` to
    /// define sub-namespaces.
    ///
    /// Both standard and custom names can also define a variant with the form
    /// `<name>/<variant>` or `<namespace>::<name>/<variant>`.
    ///
    /// All components (namespace, name, variant) must be non-empty if they are
    /// present, and must be valid identifiers (alphanumeric + underscore, not
    /// starting with a digit).
    pub fn new(name: String) -> Result<Self, Error> {
        let (main_part, variant) = if let Some(pos) = name.find('/') {
            (&name[..pos], Some(name[pos + 1..].to_string()))
        } else {
            (&*name, None)
        };

        let (namespace, base) = match main_part.rsplit_once("::") {
            Some((ns, base)) => (Some(ns.to_string()), base.to_string()),
            None => (None, main_part.to_string()),
        };

        if let Some(ref ns) = namespace {
            for component in ns.split("::") {
                if !is_valid_identifier(component) {
                    return Err(Error::InvalidParameter(format!(
                        "invalid namespace '{}' in '{}': must be a valid \
                        identifier (alphanumeric or underscore, not starting with a digit)",
                        ns, name
                    )));
                }
            }
        }

        if base.is_empty() {
            return Err(Error::InvalidParameter(format!(
                "quantity name cannot be empty in '{}'", name
            )));
        }

        if !is_valid_identifier(&base) {
            return Err(Error::InvalidParameter(format!(
                "invalid quantity name '{}' in '{}': \
                must be a valid identifier (alphanumeric or underscore, not starting with a digit)",
                base, name
            )));
        }

        if let Some(ref variant) = variant && !is_valid_identifier(variant) {
            return Err(Error::InvalidParameter(format!(
                "invalid quantity variant '{}' in '{}': \
                must be a valid identifier (alphanumeric or underscore, not starting with a digit)",
                variant, name
            )));
        }

        if namespace.is_none() && !STANDARD_QUANTITIES.contains(&&*base) {
            return Err(Error::InvalidParameter(format!(
                "'{}' is not a standard quantity name; custom quantity names must use '<namespace>::<name>'",
                name
            )));
        }

        return Ok(QuantityName {
            full: name,
            namespace,
            base,
            variant,
        })
    }

    /// Is this a custom quantity name?
    pub fn is_custom(&self) -> bool {
        self.namespace.is_some()
    }

    /// Get the base name of this quantity
    pub fn base(&self) -> &str {
        &self.base
    }

    /// Get the namespace of this quantity, if any
    pub fn namespace(&self) -> Option<&str> {
        self.namespace.as_deref()
    }

    /// Get the variant of this quantity, if any
    pub fn variant(&self) -> Option<&str> {
        self.variant.as_deref()
    }

    /// Get the full name of this quantity, including namespace and variant if
    /// present
    pub fn full(&self) -> &str {
        &self.full
    }
}

impl std::fmt::Display for QuantityName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.full())
    }
}

/// Different kind of samples a quantity can be associated with
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

impl std::fmt::Display for SampleKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SampleKind::Atom => write!(f, "atom"),
            SampleKind::AtomPair => write!(f, "atom_pair"),
            SampleKind::System => write!(f, "system"),
        }
    }
}

/// Different gradients that a quantity can have
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    pub name: QuantityName,
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
        result["name"] = value.name.full().into();
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
        let name = QuantityName::new(name.to_string())?;

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
            name: name,
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
            name: QuantityName::new("energy".into()).unwrap(),
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
        assert_eq!(parsed.name.base, "energy");
        assert_eq!(parsed.unit, "eV");
        assert_eq!(parsed.gradients, vec![Gradients::Positions]);
        assert!(matches!(parsed.sample_kind, SampleKind::Atom));
    }

    #[test]
    fn roundtrip_all_variants() {
        for sample_kind in [SampleKind::Atom, SampleKind::System, SampleKind::AtomPair] {
            for gradients in [
                vec![],
                vec![Gradients::Positions],
                vec![Gradients::Strain],
                vec![Gradients::Positions, Gradients::Strain],
            ] {
                let quantity = Quantity {
                    name: QuantityName::new("test_ns::test".into()).unwrap(),
                    unit: "unit".into(),
                    description: Some("Hello".to_string()),
                    gradients: gradients.clone(),
                    sample_kind: sample_kind,
                };
                let parsed = Quantity::try_from(&JsonValue::from(quantity.clone())).unwrap();
                assert_eq!(parsed.name, quantity.name);
                assert_eq!(parsed.unit, quantity.unit);
                assert_eq!(parsed.gradients, gradients);
                assert_eq!(parsed.sample_kind, sample_kind);
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
            QuantityName::new(name.to_string()).unwrap();
        }

        let custom = [
            "my_model::energy",
            "org::my_model::custom_qty",
            "ns1::ns2::ns3::energy",
            "some_ns::name_with_underscores",
            "_ns::_name",
        ];
        for name in custom {
            QuantityName::new(name.to_string()).unwrap();
        }

        let variants = [
            "energy/ensemble",
            "my_ns::energy/raw",
            "ns1::ns2::energy/some_variant",
        ];
        for name in variants {
            QuantityName::new(name.to_string()).unwrap();
        }

        let error = QuantityName::new(String::new()).expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: quantity name cannot be empty in ''");

        let error = QuantityName::new("not_a_standard_name".into()).expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: 'not_a_standard_name' is not a standard quantity name; custom quantity names must use '<namespace>::<name>'");

        let error = QuantityName::new("/variant".into()).expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: quantity name cannot be empty in '/variant'");

        let error = QuantityName::new("name/".into()).expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: invalid quantity variant '' in 'name/': must be a valid identifier (alphanumeric or underscore, not starting with a digit)");

        let error = QuantityName::new("::energy".into()).expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: invalid namespace '' in '::energy': must be a valid identifier (alphanumeric or underscore, not starting with a digit)");

        let error = QuantityName::new("ns::".into()).expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: quantity name cannot be empty in 'ns::'");

        let error = QuantityName::new("ns::/variant".into()).expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: quantity name cannot be empty in 'ns::/variant'");

        let error = QuantityName::new("::".into()).expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: invalid namespace '' in '::': must be a valid identifier (alphanumeric or underscore, not starting with a digit)");

        let error = QuantityName::new("123name".into()).expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: invalid quantity name '123name' in '123name': must be a valid identifier (alphanumeric or underscore, not starting with a digit)");

        let error = QuantityName::new("my_ns::123name".into()).expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: invalid quantity name '123name' in 'my_ns::123name': must be a valid identifier (alphanumeric or underscore, not starting with a digit)");

        let error = QuantityName::new("my_ns::name/123variant".into()).expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: invalid quantity variant '123variant' in 'my_ns::name/123variant': must be a valid identifier (alphanumeric or underscore, not starting with a digit)");

        let error = QuantityName::new("has spaces".into()).expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: invalid quantity name 'has spaces' in 'has spaces': must be a valid identifier (alphanumeric or underscore, not starting with a digit)");

        let error = QuantityName::new("my_ns::name/has spaces".into()).expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: invalid quantity variant 'has spaces' in 'my_ns::name/has spaces': must be a valid identifier (alphanumeric or underscore, not starting with a digit)");

        let error = QuantityName::new("has-dash".into()).expect_err("expected an error");
        assert_eq!(error.to_string(), "invalid parameter: invalid quantity name 'has-dash' in 'has-dash': must be a valid identifier (alphanumeric or underscore, not starting with a digit)");
    }
}
