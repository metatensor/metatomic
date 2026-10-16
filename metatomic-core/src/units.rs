use crate::Error;

use std::sync::LazyLock;
use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Sub};

/// Physical dimension vector with named integer exponents:
/// [Length, Time, Mass, Electric Current, Temperature]
///
/// Note: quantity of substance (mole) is intentionally not included, since we
/// want `kJ/mol` and `eV` to have the same dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Dimension {
    length: i32,
    time: i32,
    mass: i32,
    electric_current: i32,
    temperature: i32,
}

impl Dimension {
    /// Dimensionless — all exponents are zero.
    const NONE: Dimension = Dimension { length: 0, time: 0, mass: 0, electric_current: 0, temperature: 0 };

    /// Length dimension
    const LENGTH: Dimension = Dimension { length: 1, time: 0, mass: 0, electric_current: 0, temperature: 0 };
    /// Time dimension
    const TIME: Dimension = Dimension { length: 0, time: 1, mass: 0, electric_current: 0, temperature: 0 };
    /// Mass dimension
    const MASS: Dimension = Dimension { length: 0, time: 0, mass: 1, electric_current: 0, temperature: 0 };
    /// Electric charge dimension (current × time)
    const CHARGE: Dimension = Dimension { length: 0, time: 1, mass: 0, electric_current: 1, temperature: 0 };
    /// Temperature dimension
    const TEMPERATURE: Dimension = Dimension { length: 0, time: 0, mass: 0, electric_current: 0, temperature: 1 };

    /// Energy dimension: L² T⁻² M¹
    const ENERGY: Dimension = Dimension { length: 2, time: -2, mass: 1, electric_current: 0, temperature: 0 };
    /// Pressure dimension: L⁻¹ T⁻² M¹
    const PRESSURE: Dimension = Dimension { length: -1, time: -2, mass: 1, electric_current: 0, temperature: 0 };
    /// Electric dipole dimension: L¹ T¹ I¹
    const ELECTRIC_DIPOLE: Dimension = Dimension { length: 1, time: 1, mass: 0, electric_current: 1, temperature: 0 };

    fn pow(&self, p: f64) -> Dimension {
        Dimension {
            length: round_if_integer(f64::from(self.length) * p),
            time: round_if_integer(f64::from(self.time) * p),
            mass: round_if_integer(f64::from(self.mass) * p),
            electric_current: round_if_integer(f64::from(self.electric_current) * p),
            temperature: round_if_integer(f64::from(self.temperature) * p),
        }
    }
}

impl Add<&Dimension> for &Dimension {
    type Output = Dimension;

    fn add(self, other: &Dimension) -> Dimension {
        Dimension {
            length: self.length + other.length,
            time: self.time + other.time,
            mass: self.mass + other.mass,
            electric_current: self.electric_current + other.electric_current,
            temperature: self.temperature + other.temperature,
        }
    }
}

impl Sub<&Dimension> for &Dimension {
    type Output = Dimension;

    fn sub(self, other: &Dimension) -> Dimension {
        Dimension {
            length: self.length - other.length,
            time: self.time - other.time,
            mass: self.mass - other.mass,
            electric_current: self.electric_current - other.electric_current,
            temperature: self.temperature - other.temperature,
        }
    }
}

#[allow(clippy::cast_possible_truncation)]
fn round_if_integer(v: f64) -> i32 {
    let rounded = v.round();
    assert!((v - rounded).abs() <= 1e-10, "non-integer dimension exponent {} is not supported", v);
    return rounded as i32;
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use fmt::Write;
        let mut first = true;
        f.write_char('[')?;

        for (name, v) in [
            ("L", self.length),
            ("T", self.time),
            ("M", self.mass),
            ("I", self.electric_current),
            ("Θ", self.temperature),
        ] {
            if v == 0 {
                continue;
            }

            if !first {
                f.write_char(' ')?;
            }
            first = false;

            f.write_str(name)?;

            if v != 1 && v != -1 {
                write!(f, "^{}", v)?;
            }

            if v == -1 {
                f.write_str("^-1")?;
            }
        }

        if first {
            f.write_str("dimensionless")?;
        }
        f.write_char(']')?;

        Ok(())
    }
}

/// A parsed unit value: SI conversion factor and physical dimension.
#[derive(Debug, Clone)]
struct UnitValue {
    factor: f64,
    dim: Dimension,
}

/// All base units with SI factors and dimensions.
/// Factors are expressed in SI base units (m, s, kg, C, K).
/// Case-insensitive lookup: names are lowercased before searching.
static BASE_UNITS: LazyLock<HashMap<&'static str, UnitValue>> = LazyLock::new(|| {
    let mut map = HashMap::new();

    // --- Temperature ---
    map.insert("kelvin", UnitValue { factor: 1.0, dim: Dimension::TEMPERATURE });
    map.insert("k", UnitValue { factor: 1.0, dim: Dimension::TEMPERATURE });

    // --- Length ---
    map.insert("angstrom", UnitValue { factor: 1e-10, dim: Dimension::LENGTH });
    map.insert("a", UnitValue { factor: 1e-10, dim: Dimension::LENGTH });
    map.insert("bohr", UnitValue { factor: 5.2917721054482e-11, dim: Dimension::LENGTH });
    map.insert("nm", UnitValue { factor: 1e-9, dim: Dimension::LENGTH });
    map.insert("nanometer", UnitValue { factor: 1e-9, dim: Dimension::LENGTH });
    map.insert("meter", UnitValue { factor: 1.0, dim: Dimension::LENGTH });
    map.insert("m", UnitValue { factor: 1.0, dim: Dimension::LENGTH });
    map.insert("cm", UnitValue { factor: 1e-2, dim: Dimension::LENGTH });
    map.insert("centimeter", UnitValue { factor: 1e-2, dim: Dimension::LENGTH });
    map.insert("mm", UnitValue { factor: 1e-3, dim: Dimension::LENGTH });
    map.insert("millimeter", UnitValue { factor: 1e-3, dim: Dimension::LENGTH });
    map.insert("um", UnitValue { factor: 1e-6, dim: Dimension::LENGTH });
    map.insert("µm", UnitValue { factor: 1e-6, dim: Dimension::LENGTH });
    map.insert("micrometer", UnitValue { factor: 1e-6, dim: Dimension::LENGTH });

    // --- Energy ---
    map.insert("electronvolt", UnitValue { factor: 1.602176634e-19, dim: Dimension::ENERGY });
    map.insert("ev", UnitValue { factor: 1.602176634e-19, dim: Dimension::ENERGY });
    map.insert("mev", UnitValue { factor: 1.602176634e-19 * 1e-3, dim: Dimension::ENERGY });
    map.insert("hartree", UnitValue { factor: 4.359744722206048e-18, dim: Dimension::ENERGY });
    map.insert("ry", UnitValue { factor: 2.179872361103024e-18, dim: Dimension::ENERGY });
    map.insert("rydberg", UnitValue { factor: 2.179872361103024e-18, dim: Dimension::ENERGY });
    map.insert("joule", UnitValue { factor: 1.0, dim: Dimension::ENERGY });
    map.insert("j", UnitValue { factor: 1.0, dim: Dimension::ENERGY });
    map.insert("kcal", UnitValue { factor: 4184.0, dim: Dimension::ENERGY });
    map.insert("kj", UnitValue { factor: 1000.0, dim: Dimension::ENERGY });

    // --- Time ---
    map.insert("s", UnitValue { factor: 1.0, dim: Dimension::TIME });
    map.insert("second", UnitValue { factor: 1.0, dim: Dimension::TIME });
    map.insert("ms", UnitValue { factor: 1e-3, dim: Dimension::TIME });
    map.insert("millisecond", UnitValue { factor: 1e-3, dim: Dimension::TIME });
    map.insert("us", UnitValue { factor: 1e-6, dim: Dimension::TIME });
    map.insert("µs", UnitValue { factor: 1e-6, dim: Dimension::TIME });
    map.insert("microsecond", UnitValue { factor: 1e-6, dim: Dimension::TIME });
    map.insert("ns", UnitValue { factor: 1e-9, dim: Dimension::TIME });
    map.insert("nanosecond", UnitValue { factor: 1e-9, dim: Dimension::TIME });
    map.insert("ps", UnitValue { factor: 1e-12, dim: Dimension::TIME });
    map.insert("picosecond", UnitValue { factor: 1e-12, dim: Dimension::TIME });
    map.insert("fs", UnitValue { factor: 1e-15, dim: Dimension::TIME });
    map.insert("femtosecond", UnitValue { factor: 1e-15, dim: Dimension::TIME });

    // --- Mass ---
    map.insert("u", UnitValue { factor: 1.6605390689252e-27, dim: Dimension::MASS });
    map.insert("dalton", UnitValue { factor: 1.6605390689252e-27, dim: Dimension::MASS });
    map.insert("kg", UnitValue { factor: 1.0, dim: Dimension::MASS });
    map.insert("kilogram", UnitValue { factor: 1.0, dim: Dimension::MASS });
    map.insert("g", UnitValue { factor: 1e-3, dim: Dimension::MASS });
    map.insert("gram", UnitValue { factor: 1e-3, dim: Dimension::MASS });
    map.insert("electron_mass", UnitValue { factor: 9.109383713928e-31, dim: Dimension::MASS });
    map.insert("m_e", UnitValue { factor: 9.109383713928e-31, dim: Dimension::MASS });

    // --- Charge ---
    map.insert("e", UnitValue { factor: 1.602176634e-19, dim: Dimension::CHARGE });
    map.insert("coulomb", UnitValue { factor: 1.0, dim: Dimension::CHARGE });
    map.insert("c", UnitValue { factor: 1.0, dim: Dimension::CHARGE });

    // --- Pressure ---
    map.insert("pa", UnitValue { factor: 1.0, dim: Dimension::PRESSURE });
    map.insert("pascal", UnitValue { factor: 1.0, dim: Dimension::PRESSURE });
    map.insert("kpa", UnitValue { factor: 1e3, dim: Dimension::PRESSURE });
    map.insert("kilopascal", UnitValue { factor: 1e3, dim: Dimension::PRESSURE });
    map.insert("mpa", UnitValue { factor: 1e6, dim: Dimension::PRESSURE });
    map.insert("megapascal", UnitValue { factor: 1e6, dim: Dimension::PRESSURE });
    map.insert("gpa", UnitValue { factor: 1e9, dim: Dimension::PRESSURE });
    map.insert("gigapascal", UnitValue { factor: 1e9, dim: Dimension::PRESSURE });
    map.insert("bar", UnitValue { factor: 100000.0, dim: Dimension::PRESSURE });
    map.insert("atm", UnitValue { factor: 101325.0, dim: Dimension::PRESSURE });

    // --- Electric dipole moment ---
    map.insert("debye", UnitValue { factor: 1.0 / 299792458.0 * 1e-21, dim: Dimension::ELECTRIC_DIPOLE });
    map.insert("d", UnitValue { factor: 1.0 / 299792458.0 * 1e-21, dim: Dimension::ELECTRIC_DIPOLE });

    // --- Dimensionless ---
    map.insert("mol", UnitValue { factor: 6.02214076e23, dim: Dimension::NONE });

    // --- Derived ---
    map.insert("hbar", UnitValue {
        factor: 1.0545718176462e-34,
        dim: Dimension { length: 2, time: -1, mass: 1, electric_current: 0, temperature: 0 },
    });

    map
});

// ---- Tokenizer ----

#[derive(Debug, Clone)]
enum Token {
    LParen,
    RParen,
    Mul,
    Div,
    Pow,
    Value(String),
}

impl Token {
    fn precedence(&self) -> i32 {
        match self {
            Token::LParen | Token::RParen => 0,
            Token::Mul | Token::Div => 10,
            Token::Pow => 20,
            Token::Value(_) => -1,
        }
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::Mul => write!(f, "*"),
            Token::Div => write!(f, "/"),
            Token::Pow => write!(f, "^"),
            Token::Value(v) => write!(f, "{}", v),
        }
    }
}

fn tokenize(unit: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for c in unit.chars() {
        if c == '*' || c == '/' || c == '^' || c == '(' || c == ')' {
            if !current.is_empty() {
                tokens.push(Token::Value(current.clone()));
                current.clear();
            }
            let t = match c {
                '*' => Token::Mul,
                '/' => Token::Div,
                '^' => Token::Pow,
                '(' => Token::LParen,
                ')' => Token::RParen,
                _ => unreachable!(),
            };
            tokens.push(t);
        } else if !c.is_whitespace() {
            current.push(c);
        }
    }

    if !current.is_empty() {
        tokens.push(Token::Value(current));
    }

    tokens
}

// ---- Shunting-Yard ----

/// Convert infix tokens to [Reverse Polish Notation] (RPN) using the
/// [Shunting-Yard] algorithm.
///
/// RPN (also called postfix notation) writes operators after their operands,
/// e.g. `kJ / mol` becomes `kJ mol /`. This removes the need for parentheses
/// and precedence rules, making the expression easy to evaluate with a stack.
///
/// All operators are treated as left-associative.
///
/// [Reverse Polish Notation]: https://en.wikipedia.org/wiki/Reverse_Polish_notation
/// [Shunting-Yard]: https://en.wikipedia.org/wiki/Shunting-yard_algorithm
fn shunting_yard(tokens: &[Token]) -> Result<Vec<Token>, Error> {
    let mut output: Vec<Token> = Vec::new();
    let mut operators: Vec<Token> = Vec::new();

    for token in tokens {
        match token {
            Token::Value(_) => {
                output.push(token.clone());
            }
            Token::Mul | Token::Div | Token::Pow => {
                while let Some(top) = operators.last() {
                    if token.precedence() <= top.precedence() {
                        output.push(operators.pop().unwrap());
                    } else {
                        break;
                    }
                }
                operators.push(token.clone());
            }
            Token::LParen => {
                operators.push(token.clone());
            }
            Token::RParen => {
                while let Some(top) = operators.last() {
                    if matches!(top, Token::LParen) {
                        break;
                    }
                    output.push(operators.pop().unwrap());
                }
                if operators.is_empty() || !matches!(operators.last(), Some(Token::LParen)) {
                    return Err(Error::InvalidParameter(
                        "unit expression has unbalanced parentheses".into(),
                    ));
                }
                operators.pop(); // discard LParen
            }
        }
    }

    while let Some(top) = operators.pop() {
        if matches!(top, Token::LParen | Token::RParen) {
            return Err(Error::InvalidParameter(
                "unit expression has unbalanced parentheses".into(),
            ));
        }
        output.push(top);
    }

    Ok(output)
}

// ---- AST evaluator ----

struct UnitExpr {
    val: UnitExprData,
}

enum UnitExprData {
    Val(UnitValue, String),
    Mul(Box<UnitExpr>, Box<UnitExpr>),
    Div(Box<UnitExpr>, Box<UnitExpr>),
    Pow(Box<UnitExpr>, Box<UnitExpr>),
}

impl fmt::Display for UnitExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.val {
            UnitExprData::Val(_, name) => f.write_str(name),
            UnitExprData::Mul(lhs, rhs) => {
                write!(f, "({} * {})", lhs, rhs)
            }
            UnitExprData::Div(lhs, rhs) => {
                write!(f, "({} / {})", lhs, rhs)
            }
            UnitExprData::Pow(base, exponent) => {
                write!(f, "({} ^ {})", base, exponent)
            }
        }
    }
}

impl UnitExpr {
    fn eval(&self) -> Result<UnitValue, Error> {
        match &self.val {
            UnitExprData::Val(v, _) => Ok(v.clone()),
            UnitExprData::Mul(lhs, rhs) => {
                let l = lhs.eval()?;
                let r = rhs.eval()?;
                let result_factor = l.factor * r.factor;
                if !result_factor.is_finite() {
                    return Err(Error::InvalidParameter(format!(
                        "unit conversion factor overflows: multiplication result is infinite \
                         or NaN for '{}'",
                        self
                    )));
                }
                Ok(UnitValue {
                    factor: result_factor,
                    dim: &l.dim + &r.dim,
                })
            }
            UnitExprData::Div(lhs, rhs) => {
                let l = lhs.eval()?;
                let r = rhs.eval()?;
                let result_factor = l.factor / r.factor;
                if !result_factor.is_finite() {
                    return Err(Error::InvalidParameter(format!(
                        "unit conversion factor overflows: division result is infinite \
                         or NaN for '{}'",
                        self
                    )));
                }
                Ok(UnitValue {
                    factor: result_factor,
                    dim: &l.dim - &r.dim,
                })
            }
            UnitExprData::Pow(base, exponent) => {
                let b = base.eval()?;
                let e = exponent.eval()?;

                if e.dim != Dimension::NONE {
                    return Err(Error::InvalidParameter(format!(
                        "exponent in unit expression must be dimensionless, got dimension {} \
                         for exponent '{}'",
                        e.dim,
                        exponent
                    )));
                }
                let result_factor = b.factor.powf(e.factor);
                if !result_factor.is_finite() {
                    return Err(Error::InvalidParameter(format!(
                        "unit conversion factor overflows: exponentiation result is infinite \
                         or NaN for '{}'",
                        self
                    )));
                }
                Ok(UnitValue {
                    factor: result_factor,
                    dim: b.dim.pow(e.factor),
                })
            }
        }
    }
}

/// Read one expression from the [RPN] stream (recursive, pops from the back).
///
/// RPN arranges expressions as `lhs rhs op`, so `rhs` is on top of the stack
/// and must be popped first. For example `kJ mol /` pops `mol` (rhs) then
/// `kJ` (lhs) to build `Div(lhs=kJ, rhs=mol)`.
///
/// [RPN]: https://en.wikipedia.org/wiki/Reverse_Polish_notation
fn read_expr(stream: &mut Vec<Token>) -> Result<UnitExpr, Error> {
    let token = stream.pop().ok_or_else(|| {
        Error::InvalidParameter("malformed unit expression: missing a value".into())
    })?;

    match token {
        Token::Value(v) => {
            let lower = v.to_lowercase();
            if let Some(uv) = BASE_UNITS.get(lower.as_str()) {
                return Ok(UnitExpr {
                    val: UnitExprData::Val(uv.clone(), v),
                });
            }
            if let Ok(val) = v.parse::<f64>() {
                return Ok(UnitExpr {
                    val: UnitExprData::Val(UnitValue { factor: val, dim: Dimension::NONE }, v),
                });
            }
            Err(Error::InvalidParameter(format!("unknown unit '{}'", v)))
        }
        // RPN: lhs rhs Mul — pop rhs first, then lhs
        Token::Mul => {
            let rhs = read_expr(stream)?;
            let lhs = read_expr(stream)?;
            Ok(UnitExpr {
                val: UnitExprData::Mul(Box::new(lhs), Box::new(rhs)),
            })
        }
        // RPN: lhs rhs Div — pop rhs first, then lhs
        Token::Div => {
            let rhs = read_expr(stream)?;
            let lhs = read_expr(stream)?;
            Ok(UnitExpr {
                val: UnitExprData::Div(Box::new(lhs), Box::new(rhs)),
            })
        }
        // RPN: base exponent Pow — pop exponent first, then base
        Token::Pow => {
            let exponent = read_expr(stream)?;
            let base = read_expr(stream)?;
            Ok(UnitExpr {
                val: UnitExprData::Pow(Box::new(base), Box::new(exponent)),
            })
        }
        _ => Err(Error::InvalidParameter(format!(
            "unexpected symbol in unit expression: '{}'",
            token
        ))),
    }
}

/// Parse a unit expression string and return the evaluated `UnitValue`.
fn parse_unit_expression(unit: &str) -> Result<UnitValue, Error> {
    if unit.is_empty() {
        return Ok(UnitValue { factor: 1.0, dim: Dimension::NONE });
    }

    let tokens = tokenize(unit);
    if tokens.is_empty() {
        return Ok(UnitValue { factor: 1.0, dim: Dimension::NONE });
    }

    let mut rpn = shunting_yard(&tokens)?;
    let ast = read_expr(&mut rpn)?;

    if !rpn.is_empty() {
        let remaining: Vec<String> = rpn.iter().map(|t| t.to_string()).collect();
        return Err(Error::InvalidParameter(format!(
            "malformed unit expression: leftover input '{}'",
            remaining.join(" ")
        )));
    }

    ast.eval()
}

/// Get the multiplicative conversion factor to use to convert from
/// `from_unit` to `to_unit`. Both units are parsed as expressions (e.g.
/// "kJ/mol/A^2", "(eV*u)^(1/2)") and their dimensions must match.
///
/// Unit expressions are built from base units combined with `*`, `/`, `^`,
/// and parentheses. Unit lookup is case-insensitive, and whitespace is
/// ignored. For example:
///
/// - `"kJ/mol"` -- energy per mole
/// - `"eV/Angstrom^3"` -- pressure
/// - `"(eV*u)^(1/2)"` -- momentum (fractional powers)
/// - `"Hartree/Bohr"` -- force in atomic units
pub fn unit_conversion_factor(from_unit: &str, to_unit: &str) -> Result<f64, Error> {
    if from_unit.is_empty() || to_unit.is_empty() {
        return Ok(1.0);
    }

    let from = parse_unit_expression(from_unit)?;
    let to = parse_unit_expression(to_unit)?;

    if from.dim != to.dim {
        return Err(Error::InvalidParameter(format!(
            "dimension mismatch in unit conversion: '{}' has dimension {} but '{}' has dimension {}",
            from_unit,
            from.dim,
            to_unit,
            to.dim
        )));
    }

    Ok(from.factor / to.factor)
}


/// Check if a unit expression is valid and has the same dimension as the reference unit.
pub fn validate_unit(unit: &str, reference_unit: &str, context: Option<&str>) -> Result<(), Error> {
    let unit_value = parse_unit_expression(unit)?;
    let reference_value = parse_unit_expression(reference_unit)?;

    if unit_value.dim != reference_value.dim {
        return Err(Error::InvalidParameter(format!(
            "dimension mismatch{}: '{}' has dimension {} but expected dimension {}",
            context.map_or_else(String::new, |c| format!(" in {}", c)),
            unit,
            unit_value.dim,
            reference_value.dim
        )));
    }

    Ok(())
}


#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_simple() {
        let tokens = tokenize("eV");
        assert_eq!(tokens.len(), 1);
        assert!(matches!(&tokens[0], Token::Value(v) if v == "eV"));
    }

    #[test]
    fn test_tokenize_operators() {
        let tokens = tokenize("kJ/mol/A^2");
        let types: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
        assert_eq!(types, vec!["kJ", "/", "mol", "/", "A", "^", "2"]);
    }

    #[test]
    fn test_tokenize_parens() {
        let tokens = tokenize("(eV*u)^(1/2)");
        let types: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
        assert_eq!(types, vec!["(", "eV", "*", "u", ")", "^", "(", "1", "/", "2", ")"]);
    }

    #[test]
    fn test_tokenize_whitespace() {
        let tokens = tokenize("  kJ / mol ");
        let types: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
        assert_eq!(types, vec!["kJ", "/", "mol"]);
    }

    #[test]
    fn test_shunting_yard() {
        let tokens = tokenize("kJ/mol");
        let rpn = shunting_yard(&tokens).unwrap();
        let types: Vec<String> = rpn.iter().map(|t| t.to_string()).collect();
        assert_eq!(types, vec!["kJ", "mol", "/"]);

        let tokens = tokenize("kJ/mol/A^2");
        let rpn = shunting_yard(&tokens).unwrap();
        let types: Vec<String> = rpn.iter().map(|t| t.to_string()).collect();
        assert_eq!(types, vec!["kJ", "mol", "/", "A", "2", "^", "/"]);
    }

    #[test]
    fn test_parens_mismatch() {
        let tokens = tokenize("(");
        let err = shunting_yard(&tokens).expect_err("expected error");
        assert_eq!(
            err.to_string(),
            "invalid parameter: unit expression has unbalanced parentheses"
        );

        let tokens = tokenize("(eV*u");
        let err = shunting_yard(&tokens).expect_err("expected error");
        assert_eq!(
            err.to_string(),
            "invalid parameter: unit expression has unbalanced parentheses"
        );
    }

    #[test]
    fn test_simple_conversion() {
        let factor = unit_conversion_factor("eV", "eV").unwrap();
        assert_eq!(factor, 1.0);

        let factor = unit_conversion_factor("m", "A").unwrap();
        assert!((factor - 1e10).abs() < 1e-5);

        let factor = unit_conversion_factor("eV", "kJ").unwrap();
        assert!((factor - 1.602176634e-22).abs() < 1e-30);
    }

    #[test]
    fn test_dimension_mismatch() {
        let err = unit_conversion_factor("eV", "m").expect_err("expected error");
        assert_eq!(
            err.to_string(),
            "invalid parameter: dimension mismatch in unit conversion: \
             'eV' has dimension [L^2 T^-2 M] but 'm' has dimension [L]"
        );
    }

    #[test]
    fn test_empty_units() {
        let factor = unit_conversion_factor("", "").unwrap();
        assert_eq!(factor, 1.0);

        let factor = unit_conversion_factor("eV", "").unwrap();
        assert_eq!(factor, 1.0);
    }

    #[test]
    fn test_compound_units() {
        let from = unit_conversion_factor("kJ/mol", "eV").unwrap();
        assert!((from - 0.010364269656262174).abs() < 1e-15);

        let from = unit_conversion_factor("eV/A^3", "GPa").unwrap();
        assert!((from - 160.21766339999996).abs() < 1e-12);
    }

    #[test]
    fn test_case_insensitive() {
        let f1 = unit_conversion_factor("eV", "eV").unwrap();
        let f2 = unit_conversion_factor("EV", "eV").unwrap();
        assert_eq!(f1, f2);

        let factor = unit_conversion_factor("eV", "MeV").unwrap();
        assert!((factor - 1000.0).abs() < 1e-12);
    }

    #[test]
    fn test_unknown_unit() {
        let err = unit_conversion_factor("foo", "eV").expect_err("expected error");
        assert_eq!(err.to_string(), "invalid parameter: unknown unit 'foo'");
    }

    #[test]
    fn test_numeric_literal() {
        let factor = unit_conversion_factor("2", "1").unwrap();
        assert_eq!(factor, 2.0);
    }

    #[test]
    fn test_fractional_power() {
        let err = unit_conversion_factor("(eV*u)^(1/2)", "eV*u").expect_err("expected error");
        assert_eq!(
            err.to_string(),
            "invalid parameter: dimension mismatch in unit conversion: \
             '(eV*u)^(1/2)' has dimension [L T^-1 M] but 'eV*u' has dimension [L^2 T^-2 M^2]"
        );

        let factor = unit_conversion_factor("(eV*u)^(1/2)", "(eV*u)^(1/2)").unwrap();
        assert_eq!(factor, 1.0);
    }

    #[test]
    fn test_dimension_to_string() {
        assert_eq!(Dimension::NONE.to_string(), "[dimensionless]");
        assert_eq!(Dimension::LENGTH.to_string(), "[L]");
        assert_eq!(Dimension::ENERGY.to_string(), "[L^2 T^-2 M]");
        assert_eq!(Dimension::PRESSURE.to_string(), "[L^-1 T^-2 M]");
        assert_eq!(Dimension::TEMPERATURE.to_string(), "[Θ]");

        let velocity = Dimension { length: 1, time: -1, mass: 0, electric_current: 0, temperature: 0 };
        assert_eq!(velocity.to_string(), "[L T^-1]");
    }
}
