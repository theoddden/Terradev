use percent_encoding::{percent_encode, NON_ALPHANUMERIC};

pub fn percent_encode_rfc3986(input: &str) -> String {
    percent_encode(input.as_bytes(), NON_ALPHANUMERIC).to_string()
}

pub fn encode_parameters(params: &[(String, String)]) -> String {
    params
        .iter()
        .map(|(key, value)| {
            format!("{}={}", percent_encode_rfc3986(key), percent_encode_rfc3986(value))
        })
        .collect::<Vec<_>>()
        .join("&")
}
