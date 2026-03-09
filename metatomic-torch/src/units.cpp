#include <cctype>
#include <cmath>

#include <array>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <torch/torch.h>

#include "metatomic/torch/units.hpp"

/******************************************************************************/
/*** Unit expression parser with SI-based dimensional analysis             ***/
/******************************************************************************/

/// Physical dimension vector: [Length, Time, Mass, Charge, Temperature]
/// Exponents are double to support fractional powers like (eV*u)^(1/2).
struct Dimension {
    std::array<double, 5> exponents = {};

    Dimension operator*(const Dimension& other) const {
        Dimension result;
        for (size_t i = 0; i < 5; ++i) {
            result.exponents[i] = exponents[i] + other.exponents[i];
        }
        return result;
    }

    Dimension operator/(const Dimension& other) const {
        Dimension result;
        for (size_t i = 0; i < 5; ++i) {
            result.exponents[i] = exponents[i] - other.exponents[i];
        }
        return result;
    }

    Dimension pow(double p) const {
        Dimension result;
        for (size_t i = 0; i < 5; ++i) {
            result.exponents[i] = exponents[i] * p;
        }
        return result;
    }

    bool operator==(const Dimension& other) const {
        for (size_t i = 0; i < 5; ++i) {
            if (std::fabs(exponents[i] - other.exponents[i]) > 1e-10) {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const Dimension& other) const {
        return !(*this == other);
    }

    std::string to_string() const {
        static const char* names[] = {"L", "T", "M", "Q", "Th"};
        std::string result = "[";
        for (size_t i = 0; i < 5; ++i) {
            if (i > 0) result += ",";
            result += names[i];
            result += "=";
            // format as integer if close to integer, else as decimal
            double v = exponents[i];
            if (std::fabs(v - std::round(v)) < 1e-10) {
                result += std::to_string(static_cast<int>(std::round(v)));
            } else {
                result += std::to_string(v);
            }
        }
        result += "]";
        return result;
    }
};

/// A parsed unit value: SI conversion factor and physical dimension.
struct UnitValue {
    double factor;
    Dimension dim;
};

// Dimension constants for readability
//                                         L   T   M   Q   Th
static const Dimension DIM_LENGTH      = {{  1,  0,  0,  0,  0 }};
static const Dimension DIM_TIME        = {{  0,  1,  0,  0,  0 }};
static const Dimension DIM_MASS        = {{  0,  0,  1,  0,  0 }};
static const Dimension DIM_CHARGE      = {{  0,  0,  0,  1,  0 }};
static const Dimension DIM_TEMPERATURE = {{  0,  0,  0,  0,  1 }};
static const Dimension DIM_ENERGY      = {{  2, -2,  1,  0,  0 }};
static const Dimension DIM_NONE        = {{  0,  0,  0,  0,  0 }};

/// Lowercase a string in place and return it.
static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c) {
            // Only lowercase ASCII; preserve UTF-8 continuation bytes
            return c < 128 ? static_cast<char>(std::tolower(c)) : static_cast<char>(c);
        }
    );
    return s;
}

/// All base units with SI factors and dimensions.
/// Factors are expressed in SI base units (m, s, kg, C, K).
/// Case-insensitive lookup: tokens are lowercased before searching.
static const std::unordered_map<std::string, UnitValue>& base_units() {
    static const auto units = std::unordered_map<std::string, UnitValue>{
        // --- Length ---
        {"angstrom", {1e-10, DIM_LENGTH}},
        {"a",        {1e-10, DIM_LENGTH}},
        {"bohr",     {5.29177210903e-11, DIM_LENGTH}},
        {"nm",       {1e-9, DIM_LENGTH}},
        {"nanometer",{1e-9, DIM_LENGTH}},
        {"meter",    {1.0, DIM_LENGTH}},
        {"m",        {1.0, DIM_LENGTH}},
        {"cm",       {1e-2, DIM_LENGTH}},
        {"centimeter",{1e-2, DIM_LENGTH}},
        {"mm",       {1e-3, DIM_LENGTH}},
        {"millimeter",{1e-3, DIM_LENGTH}},
        {"um",       {1e-6, DIM_LENGTH}},
        {"µm", {1e-6, DIM_LENGTH}},
        {"micrometer",{1e-6, DIM_LENGTH}},

        // --- Energy ---
        {"ev",       {1.602176634e-19, DIM_ENERGY}},
        {"mev",      {1.602176634e-22, DIM_ENERGY}},
        {"hartree",  {4.3597447222071e-18, DIM_ENERGY}},
        {"ry",       {2.1798723611e-18, DIM_ENERGY}},
        {"rydberg",  {2.1798723611e-18, DIM_ENERGY}},
        {"joule",    {1.0, DIM_ENERGY}},
        {"j",        {1.0, DIM_ENERGY}},
        {"kcal",     {4184.0, DIM_ENERGY}},
        {"kj",       {1000.0, DIM_ENERGY}},

        // --- Time ---
        {"s",        {1.0, DIM_TIME}},
        {"second",   {1.0, DIM_TIME}},
        {"ms",       {1e-3, DIM_TIME}},
        {"millisecond", {1e-3, DIM_TIME}},
        {"us",       {1e-6, DIM_TIME}},
        {"µs", {1e-6, DIM_TIME}},
        {"microsecond", {1e-6, DIM_TIME}},
        {"ns",       {1e-9, DIM_TIME}},
        {"nanosecond",{1e-9, DIM_TIME}},
        {"ps",       {1e-12, DIM_TIME}},
        {"picosecond",{1e-12, DIM_TIME}},
        {"fs",       {1e-15, DIM_TIME}},
        {"femtosecond",{1e-15, DIM_TIME}},

        // --- Mass ---
        {"u",        {1.66053906660e-27, DIM_MASS}},
        {"dalton",   {1.66053906660e-27, DIM_MASS}},
        {"kg",       {1.0, DIM_MASS}},
        {"kilogram", {1.0, DIM_MASS}},
        {"g",        {1e-3, DIM_MASS}},
        {"gram",     {1e-3, DIM_MASS}},
        {"electron_mass", {9.1093837015e-31, DIM_MASS}},
        {"m_e",      {9.1093837015e-31, DIM_MASS}},

        // --- Charge ---
        {"e",        {1.602176634e-19, DIM_CHARGE}},
        {"coulomb",  {1.0, DIM_CHARGE}},
        {"c",        {1.0, DIM_CHARGE}},

        // --- Dimensionless ---
        {"mol",      {6.02214076e23, DIM_NONE}},

        // --- Derived ---
        {"hbar",     {1.054571817e-34, {{2, -1, 1, 0, 0}}}},
    };
    return units;
}

// ---- Tokenizer ----

enum class TokenType {
    LParen, RParen, Mul, Div, Pow, Value
};

struct Token {
    TokenType type;
    std::string value; // only meaningful for Value tokens

    int precedence() const {
        switch (type) {
            case TokenType::LParen:
            case TokenType::RParen: return 0;
            case TokenType::Mul:
            case TokenType::Div: return 10;
            case TokenType::Pow: return 20;
            default: return -1;
        }
    }

    std::string as_str() const {
        switch (type) {
            case TokenType::LParen: return "(";
            case TokenType::RParen: return ")";
            case TokenType::Mul: return "*";
            case TokenType::Div: return "/";
            case TokenType::Pow: return "^";
            case TokenType::Value: return value;
        }
        return "?";
    }
};

static std::vector<Token> tokenize(const std::string& unit) {
    std::vector<Token> tokens;
    std::string current;

    for (size_t i = 0; i < unit.size(); ++i) {
        auto byte = static_cast<unsigned char>(unit[i]);

        // Handle UTF-8 micro sign (U+00B5): 0xC2 0xB5
        if (byte == 0xC2 && i + 1 < unit.size()
            && static_cast<unsigned char>(unit[i + 1]) == 0xB5)
        {
            current += unit[i];
            current += unit[i + 1];
            ++i; // skip second byte
            continue;
        }

        char ch = unit[i];
        if (ch == '*' || ch == '/' || ch == '^' || ch == '(' || ch == ')') {
            if (!current.empty()) {
                tokens.push_back({TokenType::Value, current});
                current.clear();
            }
            TokenType t;
            switch (ch) {
                case '*': t = TokenType::Mul; break;
                case '/': t = TokenType::Div; break;
                case '^': t = TokenType::Pow; break;
                case '(': t = TokenType::LParen; break;
                case ')': t = TokenType::RParen; break;
                default: t = TokenType::Value; break; // unreachable
            }
            tokens.push_back({t, std::string(1, ch)});
        } else if (!std::isspace(byte)) {
            current += ch;
        }
    }
    if (!current.empty()) {
        tokens.push_back({TokenType::Value, current});
    }
    return tokens;
}

// ---- Shunting-Yard ----

/// Convert infix tokens to Reverse Polish Notation using the Shunting-Yard
/// algorithm. All operators are treated as left-associative.
static std::vector<Token> shunting_yard(const std::vector<Token>& tokens) {
    std::vector<Token> output;
    std::vector<Token> operators;

    for (const auto& token : tokens) {
        switch (token.type) {
            case TokenType::Value:
                output.push_back(token);
                break;
            case TokenType::Mul:
            case TokenType::Div:
            case TokenType::Pow: {
                while (!operators.empty()) {
                    const auto& top = operators.back();
                    // left-associative: pop while top >= current
                    if (token.precedence() <= top.precedence()) {
                        output.push_back(operators.back());
                        operators.pop_back();
                    } else {
                        break;
                    }
                }
                operators.push_back(token);
                break;
            }
            case TokenType::LParen:
                operators.push_back(token);
                break;
            case TokenType::RParen: {
                while (!operators.empty() && operators.back().type != TokenType::LParen) {
                    output.push_back(operators.back());
                    operators.pop_back();
                }
                if (operators.empty() || operators.back().type != TokenType::LParen) {
                    C10_THROW_ERROR(ValueError,
                        "unit expression has unbalanced parentheses"
                    );
                }
                operators.pop_back(); // discard LParen
                break;
            }
        }
    }

    while (!operators.empty()) {
        if (operators.back().type == TokenType::LParen ||
            operators.back().type == TokenType::RParen) {
            C10_THROW_ERROR(ValueError,
                "unit expression has unbalanced parentheses"
            );
        }
        output.push_back(operators.back());
        operators.pop_back();
    }

    return output;
}

// ---- AST evaluator ----
//
// Departure from lumol: Pow exponent is a full sub-expression (not just i32)
// to handle ^(1/2). The exponent sub-expression must be dimensionless; its
// factor value becomes the exponent.

struct UnitExpr;
using UnitExprPtr = std::unique_ptr<UnitExpr>;

struct UnitExpr {
    struct Val { UnitValue value; };
    struct Mul { UnitExprPtr lhs, rhs; };
    struct Div { UnitExprPtr lhs, rhs; };
    struct Pow { UnitExprPtr base, exponent; };

    std::variant<Val, Mul, Div, Pow> data;

    UnitValue eval() const {
        return std::visit([](const auto& v) -> UnitValue {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, Val>) {
                return v.value;
            } else if constexpr (std::is_same_v<T, Mul>) {
                auto l = v.lhs->eval();
                auto r = v.rhs->eval();
                return {l.factor * r.factor, l.dim * r.dim};
            } else if constexpr (std::is_same_v<T, Div>) {
                auto l = v.lhs->eval();
                auto r = v.rhs->eval();
                return {l.factor / r.factor, l.dim / r.dim};
            } else if constexpr (std::is_same_v<T, Pow>) {
                auto b = v.base->eval();
                auto e = v.exponent->eval();
                if (e.dim != DIM_NONE) {
                    C10_THROW_ERROR(ValueError,
                        "exponent in unit expression must be dimensionless, "
                        "got dimension " + e.dim.to_string()
                    );
                }
                return {std::pow(b.factor, e.factor), b.dim.pow(e.factor)};
            }
        }, data);
    }
};

/// Read one expression from the RPN stream (recursive, pops from the back).
static UnitExprPtr read_expr(std::vector<Token>& stream) {
    if (stream.empty()) {
        C10_THROW_ERROR(ValueError,
            "malformed unit expression: missing a value"
        );
    }

    auto token = stream.back();
    stream.pop_back();

    switch (token.type) {
        case TokenType::Value: {
            auto lower = to_lower(token.value);
            const auto& units = base_units();
            auto it = units.find(lower);
            if (it != units.end()) {
                auto expr = std::make_unique<UnitExpr>();
                expr->data = UnitExpr::Val{it->second};
                return expr;
            }
            // try parsing as a numeric literal (dimensionless)
            try {
                double val = std::stod(token.value);
                auto expr = std::make_unique<UnitExpr>();
                expr->data = UnitExpr::Val{{val, DIM_NONE}};
                return expr;
            } catch (...) {
                C10_THROW_ERROR(ValueError,
                    "unknown unit '" + token.value + "'"
                );
            }
        }
        case TokenType::Mul: {
            auto rhs = read_expr(stream);
            auto lhs = read_expr(stream);
            auto expr = std::make_unique<UnitExpr>();
            expr->data = UnitExpr::Mul{std::move(lhs), std::move(rhs)};
            return expr;
        }
        case TokenType::Div: {
            auto rhs = read_expr(stream);
            auto lhs = read_expr(stream);
            auto expr = std::make_unique<UnitExpr>();
            expr->data = UnitExpr::Div{std::move(lhs), std::move(rhs)};
            return expr;
        }
        case TokenType::Pow: {
            // Exponent is a full sub-expression (supports ^(1/2))
            auto exponent = read_expr(stream);
            auto base = read_expr(stream);
            auto expr = std::make_unique<UnitExpr>();
            expr->data = UnitExpr::Pow{std::move(base), std::move(exponent)};
            return expr;
        }
        default:
            C10_THROW_ERROR(ValueError,
                "unexpected token in unit expression: " + token.as_str()
            );
    }
}

/// Parse a unit expression string and return the evaluated UnitValue.
static UnitValue parse_unit_expression(const std::string& unit) {
    if (unit.empty()) {
        return {1.0, DIM_NONE};
    }

    auto tokens = tokenize(unit);
    if (tokens.empty()) {
        return {1.0, DIM_NONE};
    }

    auto rpn = shunting_yard(tokens);
    auto ast = read_expr(rpn);

    if (!rpn.empty()) {
        std::string remaining;
        for (const auto& t : rpn) {
            if (!remaining.empty()) remaining += " ";
            remaining += t.as_str();
        }
        C10_THROW_ERROR(ValueError,
            "malformed unit expression: leftover tokens '" + remaining + "'"
        );
    }

    return ast->eval();
}

// ---- Quantity dimension map (for validate_unit) ----

static const std::unordered_map<std::string, Dimension>& quantity_dims() {
    static const auto dims = std::unordered_map<std::string, Dimension>{
        {"length",   DIM_LENGTH},
        {"energy",   DIM_ENERGY},
        {"force",    {{1, -2, 1, 0, 0}}},   // energy/length
        {"pressure", {{-1, -2, 1, 0, 0}}},  // energy/length^3
        {"momentum", {{1, -1, 1, 0, 0}}},   // mass*length/time
        {"mass",     DIM_MASS},
        {"velocity", {{1, -1, 0, 0, 0}}},   // length/time
        {"charge",   DIM_CHARGE},
    };
    return dims;
}

// ---- Public API ----

/// 2-argument unit_conversion_factor: parse both expressions, check dimensions
/// match, and return from_factor / to_factor.
double metatomic_torch::unit_conversion_factor(
    const std::string& from_unit,
    const std::string& to_unit
) {
    if (from_unit.empty() || to_unit.empty()) {
        return 1.0;
    }

    auto from = parse_unit_expression(from_unit);
    auto to = parse_unit_expression(to_unit);

    if (from.dim != to.dim) {
        C10_THROW_ERROR(ValueError,
            "dimension mismatch in unit conversion: '" + from_unit
            + "' has dimension " + from.dim.to_string()
            + " but '" + to_unit + "' has dimension " + to.dim.to_string()
        );
    }

    return from.factor / to.factor;
}

bool metatomic_torch::valid_quantity(const std::string& quantity) {
    if (quantity.empty()) {
        return false;
    }

    const auto& dims = quantity_dims();
    if (dims.find(quantity) == dims.end()) {
        auto valid_quantities = std::vector<std::string>();
        for (const auto& it: dims) {
            valid_quantities.emplace_back(it.first);
        }
        std::sort(valid_quantities.begin(), valid_quantities.end());

        static std::unordered_set<std::string> ALREADY_WARNED = {};
        if (ALREADY_WARNED.insert(quantity).second) {
            TORCH_WARN(
                "unknown quantity '", quantity, "', only [",
                torch::str(valid_quantities), "] are supported"
            );
        }
        return false;
    } else {
        return true;
    }
}


void metatomic_torch::validate_unit(const std::string& quantity, const std::string& unit) {
    if (quantity.empty() || unit.empty()) {
        return;
    }

    // Always try to parse the expression (catches syntax errors)
    auto parsed = parse_unit_expression(unit);

    // If the quantity is known, verify dimensions match
    const auto& dims = quantity_dims();
    auto it = dims.find(quantity);
    if (it != dims.end()) {
        if (parsed.dim != it->second) {
            C10_THROW_ERROR(ValueError,
                "unit '" + unit + "' has dimension " + parsed.dim.to_string()
                + " which is incompatible with quantity '" + quantity
                + "' (expected " + it->second.to_string() + ")"
            );
        }
    }
}

/// Deprecated 3-argument overload: ignores quantity, delegates to 2-arg.
double metatomic_torch::unit_conversion_factor(
    const std::string& quantity,
    const std::string& from_unit,
    const std::string& to_unit
) {
    static std::once_flag warn_flag;
    std::call_once(warn_flag, [&]() {
        TORCH_WARN(
            "the 3-argument unit_conversion_factor(quantity, from, to) is "
            "deprecated; use the 2-argument unit_conversion_factor(from, to) "
            "instead. The quantity parameter is no longer needed."
        );
    });

    return metatomic_torch::unit_conversion_factor(from_unit, to_unit);
}
