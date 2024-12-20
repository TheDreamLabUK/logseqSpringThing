use regex::Regex;
use lazy_static::lazy_static;

lazy_static! {
    static ref CAMEL_TO_SNAKE: Regex = Regex::new(r"([a-z0-9])([A-Z])").unwrap();
    static ref SNAKE_TO_CAMEL: Regex = Regex::new(r"[_-]([a-z])").unwrap();
    static ref CAMEL_TO_KEBAB: Regex = Regex::new(r"([a-z0-9])([A-Z])").unwrap();
    static ref SNAKE_TO_KEBAB: Regex = Regex::new(r"_").unwrap();
}

pub fn to_snake_case(s: &str) -> String {
    let s = s.replace('-', "_");
    CAMEL_TO_SNAKE.replace_all(&s, "${1}_${2}").to_lowercase()
}

pub fn to_camel_case(s: &str) -> String {
    let s = SNAKE_TO_CAMEL.replace_all(s, |caps: &regex::Captures| {
        caps[1].to_uppercase()
    }).to_string();
    
    // Handle the first character
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_lowercase().collect::<String>() + chars.as_str(),
    }
}

pub fn to_kebab_case(s: &str) -> String {
    // First convert to snake case
    let snake = to_snake_case(s);
    // Then replace underscores with hyphens
    SNAKE_TO_KEBAB.replace_all(&snake, "-").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_snake_case() {
        assert_eq!(to_snake_case("baseSize"), "base_size");
        assert_eq!(to_snake_case("enableHoverEffect"), "enable_hover_effect");
        assert_eq!(to_snake_case("backgroundColor"), "background_color");
        assert_eq!(to_snake_case("base-size"), "base_size");
        assert_eq!(to_snake_case("enable-hover-effect"), "enable_hover_effect");
    }

    #[test]
    fn test_to_camel_case() {
        assert_eq!(to_camel_case("base_size"), "baseSize");
        assert_eq!(to_camel_case("enable_hover_effect"), "enableHoverEffect");
        assert_eq!(to_camel_case("background_color"), "backgroundColor");
        assert_eq!(to_camel_case("base-size"), "baseSize");
        assert_eq!(to_camel_case("enable-hover-effect"), "enableHoverEffect");
    }

    #[test]
    fn test_to_kebab_case() {
        assert_eq!(to_kebab_case("baseSize"), "base-size");
        assert_eq!(to_kebab_case("enableHoverEffect"), "enable-hover-effect");
        assert_eq!(to_kebab_case("backgroundColor"), "background-color");
        assert_eq!(to_kebab_case("base_size"), "base-size");
        assert_eq!(to_kebab_case("enable_hover_effect"), "enable-hover-effect");
    }
}
