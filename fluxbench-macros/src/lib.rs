//! FluxBench Macros
//!
//! Procedural macros for benchmark registration and async wrapping.
//!
//! ## Macros
//!
//! - `#[flux::bench]` - Register a benchmark function
//! - `#[flux::verify]` - Define a performance assertion
//! - `#[flux::synthetic]` - Define a computed metric
//! - `#[flux::group]` - Group related benchmarks with metadata
//! - `#[flux::report]` - Define a dashboard layout

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{ItemFn, ItemStruct, parse_macro_input};

// ============================================================================
// Attribute Parsing Helpers
// ============================================================================
//
// Common utilities for parsing macro attributes, eliminating duplication
// across the 6 procedural macros.

mod attr {
    use syn::meta::ParseNestedMeta;

    /// Get the attribute name as a string
    pub fn name(meta: &ParseNestedMeta) -> String {
        meta.path
            .get_ident()
            .map(|i| i.to_string())
            .unwrap_or_default()
    }

    /// Parse a string literal attribute: `attr = "value"`
    pub fn string(meta: &ParseNestedMeta) -> syn::Result<String> {
        let value: syn::LitStr = meta.value()?.parse()?;
        Ok(value.value())
    }

    /// Parse a float literal attribute: `attr = 1.5`
    pub fn float(meta: &ParseNestedMeta) -> syn::Result<f64> {
        let value: syn::LitFloat = meta.value()?.parse()?;
        value.base10_parse()
    }

    /// Parse an integer literal attribute: `attr = 42`
    pub fn int<T>(meta: &ParseNestedMeta) -> syn::Result<Option<T>>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display,
    {
        let value: syn::LitInt = meta.value()?.parse()?;
        Ok(value.base10_parse().ok())
    }

    /// Parse a boolean literal attribute: `attr = true`
    pub fn bool(meta: &ParseNestedMeta) -> syn::Result<bool> {
        let value: syn::LitBool = meta.value()?.parse()?;
        Ok(value.value())
    }

    /// Parse a comma-separated string as tags: `tags = "a, b, c"`
    pub fn tags(meta: &ParseNestedMeta) -> syn::Result<Vec<String>> {
        let value: syn::LitStr = meta.value()?.parse()?;
        Ok(value
            .value()
            .split(',')
            .map(|s| s.trim().to_string())
            .collect())
    }

    /// Parse a bracketed array of strings: `items = ["a", "b", "c"]`
    pub fn string_array(meta: &ParseNestedMeta) -> syn::Result<Vec<String>> {
        meta.value()?;
        let content;
        syn::bracketed!(content in meta.input);
        let items: syn::punctuated::Punctuated<syn::LitStr, syn::Token![,]> =
            syn::punctuated::Punctuated::parse_terminated(&content)?;
        Ok(items.iter().map(|s| s.value()).collect())
    }

    /// Create an unknown attribute error
    pub fn unknown(meta: &ParseNestedMeta, name: &str) -> syn::Error {
        meta.error(format!("unknown attribute: {}", name))
    }

    /// Skip/ignore an attribute (parse but discard)
    pub fn skip_string(meta: &ParseNestedMeta) -> syn::Result<()> {
        let _: syn::LitStr = meta.value()?.parse()?;
        Ok(())
    }

    /// Skip/ignore an integer attribute
    pub fn skip_int(meta: &ParseNestedMeta) -> syn::Result<()> {
        let _: syn::LitInt = meta.value()?.parse()?;
        Ok(())
    }
}

/// Register a benchmark function
///
/// # Example
///
/// ```ignore
/// #[flux::bench]
/// fn my_benchmark(b: &mut Bencher) {
///     b.iter(|| expensive_operation());
/// }
///
/// // With configuration
/// #[flux::bench(
///     id = "custom_id",
///     group = "parsing",
///     severity = "critical",
///     threshold = 5.0,
///     budget = "100ms"
/// )]
/// fn critical_benchmark(b: &mut Bencher) { ... }
///
/// // Async with multi-threaded runtime
/// #[flux::bench(runtime = "multi_thread", worker_threads = 4)]
/// async fn async_benchmark(b: &mut Bencher) { ... }
/// ```
#[proc_macro_attribute]
pub fn bench(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = TokenStream2::from(args);
    let func = parse_macro_input!(item as ItemFn);

    bench_impl(args, func)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

fn bench_impl(args: TokenStream2, func: ItemFn) -> Result<TokenStream2, syn::Error> {
    // Validate signature
    validate_signature(&func)?;

    // Parse configuration
    let config = parse_bench_config(args)?;

    // Generate identifiers
    let fn_name = &func.sig.ident;
    let fn_name_str = fn_name.to_string();
    let wrapper_name = format_ident!("_flux_wrapper_{}", fn_name);

    // Handle async functions
    let is_async = func.sig.asyncness.is_some();
    let runner_block = if is_async {
        generate_async_runner(&config, fn_name)
    } else {
        quote! {
            #fn_name(bencher);
        }
    };

    // Config values
    let id = config.id.unwrap_or_else(|| fn_name_str.clone());
    let group = config.group.unwrap_or_else(|| "default".to_string());
    let severity = match config.severity.as_deref() {
        Some("critical") => quote! { ::fluxbench::Severity::Critical },
        Some("warning") => quote! { ::fluxbench::Severity::Warning },
        _ => quote! { ::fluxbench::Severity::Info },
    };
    let threshold = config.threshold.unwrap_or(0.0);
    let budget_ns = config
        .budget_ns
        .map(|v| quote! { Some(#v) })
        .unwrap_or(quote! { None });
    let tags: Vec<_> = config.tags.iter().map(|t| quote! { #t }).collect();

    Ok(quote! {
        #func

        #[doc(hidden)]
        #[allow(non_snake_case)]
        fn #wrapper_name(bencher: &mut ::fluxbench::Bencher) {
            #runner_block
        }

        ::fluxbench::internal::inventory::submit! {
            ::fluxbench::BenchmarkDef {
                id: #id,
                name: #fn_name_str,
                group: #group,
                severity: #severity,
                threshold: #threshold,
                budget_ns: #budget_ns,
                tags: &[#(#tags),*],
                runner_fn: #wrapper_name,
                file: file!(),
                line: line!(),
                module_path: module_path!(),
            }
        }
    })
}

fn generate_async_runner(config: &BenchConfig, fn_name: &syn::Ident) -> TokenStream2 {
    let (runtime_builder, runtime_config) = match &config.async_runtime {
        AsyncRuntimeConfig::CurrentThread {
            enable_time,
            enable_io,
        } => {
            let time = if *enable_time {
                quote! { .enable_time() }
            } else {
                quote! {}
            };
            let io = if *enable_io {
                quote! { .enable_io() }
            } else {
                quote! {}
            };
            (quote! { new_current_thread() }, quote! { #time #io })
        }
        AsyncRuntimeConfig::MultiThread {
            worker_threads,
            enable_time,
            enable_io,
        } => {
            let workers = worker_threads
                .map(|n| quote! { .worker_threads(#n) })
                .unwrap_or(quote! {});
            let time = if *enable_time {
                quote! { .enable_time() }
            } else {
                quote! {}
            };
            let io = if *enable_io {
                quote! { .enable_io() }
            } else {
                quote! {}
            };
            (quote! { new_multi_thread() }, quote! { #workers #time #io })
        }
    };

    quote! {
        let rt = ::fluxbench::internal::tokio::runtime::Builder::#runtime_builder
            #runtime_config
            .build()
            .expect("FluxBench: Failed to create async runtime");

        rt.block_on(async {
            #fn_name(bencher).await;
        });
    }
}

#[derive(Debug, Clone)]
enum AsyncRuntimeConfig {
    CurrentThread {
        enable_time: bool,
        enable_io: bool,
    },
    MultiThread {
        worker_threads: Option<usize>,
        enable_time: bool,
        enable_io: bool,
    },
}

impl Default for AsyncRuntimeConfig {
    fn default() -> Self {
        AsyncRuntimeConfig::CurrentThread {
            enable_time: true,
            enable_io: true,
        }
    }
}

#[derive(Default)]
struct BenchConfig {
    id: Option<String>,
    group: Option<String>,
    severity: Option<String>,
    threshold: Option<f64>,
    budget_ns: Option<u64>,
    tags: Vec<String>,
    async_runtime: AsyncRuntimeConfig,
}

fn parse_bench_config(args: TokenStream2) -> Result<BenchConfig, syn::Error> {
    let mut config = BenchConfig::default();
    let mut runtime_type: Option<String> = None;
    let mut worker_threads: Option<usize> = None;
    let mut enable_time = true;
    let mut enable_io = true;

    if args.is_empty() {
        return Ok(config);
    }

    let parser = syn::meta::parser(|meta| {
        let name = attr::name(&meta);
        match name.as_str() {
            "id" => config.id = Some(attr::string(&meta)?),
            "group" => config.group = Some(attr::string(&meta)?),
            "severity" => config.severity = Some(attr::string(&meta)?),
            "threshold" => config.threshold = Some(attr::float(&meta)?),
            "budget" => config.budget_ns = parse_duration(&attr::string(&meta)?),
            "runtime" => runtime_type = Some(attr::string(&meta)?),
            "worker_threads" => worker_threads = attr::int(&meta)?,
            "enable_time" => enable_time = attr::bool(&meta)?,
            "enable_io" => enable_io = attr::bool(&meta)?,
            "tags" => config.tags = attr::tags(&meta)?,
            "iterations" => attr::skip_int(&meta)?,
            _ => return Err(attr::unknown(&meta, &name)),
        }
        Ok(())
    });

    syn::parse::Parser::parse2(parser, args)?;

    config.async_runtime = match runtime_type.as_deref() {
        Some("multi_thread") | Some("multi-thread") => AsyncRuntimeConfig::MultiThread {
            worker_threads,
            enable_time,
            enable_io,
        },
        _ => AsyncRuntimeConfig::CurrentThread {
            enable_time,
            enable_io,
        },
    };

    Ok(config)
}

fn validate_signature(func: &ItemFn) -> syn::Result<()> {
    if func.sig.inputs.len() != 1 {
        return Err(syn::Error::new_spanned(
            &func.sig,
            "FluxBench: Function must take exactly one argument: `&mut Bencher`",
        ));
    }
    Ok(())
}

fn parse_duration(s: &str) -> Option<u64> {
    let s = s.trim();
    if let Some(ms) = s.strip_suffix("ms") {
        ms.trim().parse::<u64>().ok().map(|v| v * 1_000_000)
    } else if let Some(us) = s.strip_suffix("us").or_else(|| s.strip_suffix("µs")) {
        us.trim().parse::<u64>().ok().map(|v| v * 1_000)
    } else if let Some(ns) = s.strip_suffix("ns") {
        ns.trim().parse::<u64>().ok()
    } else if let Some(s_val) = s.strip_suffix('s') {
        s_val.trim().parse::<u64>().ok().map(|v| v * 1_000_000_000)
    } else {
        None
    }
}

/// Define a performance verification
#[proc_macro_attribute]
pub fn verify(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = TokenStream2::from(args);
    let input = parse_macro_input!(item as ItemStruct);

    verify_impl(args, input)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

fn verify_impl(args: TokenStream2, input: ItemStruct) -> Result<TokenStream2, syn::Error> {
    let struct_name = &input.ident;
    let struct_name_str = struct_name.to_string();

    let mut expr = String::new();
    let mut severity = quote! { ::fluxbench::Severity::Critical };
    let mut margin = 0.0f64;

    let parser = syn::meta::parser(|meta| {
        let name = attr::name(&meta);
        match name.as_str() {
            "expr" => expr = attr::string(&meta)?,
            "severity" => {
                severity = match attr::string(&meta)?.as_str() {
                    "critical" => quote! { ::fluxbench::Severity::Critical },
                    "warning" => quote! { ::fluxbench::Severity::Warning },
                    _ => quote! { ::fluxbench::Severity::Info },
                };
            }
            "margin" => margin = attr::float(&meta)?,
            "bench" => attr::skip_string(&meta)?,
            _ => return Err(attr::unknown(&meta, &name)),
        }
        Ok(())
    });

    syn::parse::Parser::parse2(parser, args)?;

    Ok(quote! {
        #input

        ::fluxbench::internal::inventory::submit! {
            ::fluxbench::VerifyDef {
                id: #struct_name_str,
                expression: #expr,
                severity: #severity,
                margin: #margin,
            }
        }
    })
}

/// Define a synthetic (computed) metric
#[proc_macro_attribute]
pub fn synthetic(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = TokenStream2::from(args);
    let input = parse_macro_input!(item as ItemStruct);

    synthetic_impl(args, input)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

fn synthetic_impl(args: TokenStream2, input: ItemStruct) -> Result<TokenStream2, syn::Error> {
    let struct_name = &input.ident;
    let struct_name_str = struct_name.to_string();

    let mut id = struct_name_str.clone();
    let mut formula = String::new();
    let mut unit: Option<String> = None;

    let parser = syn::meta::parser(|meta| {
        let name = attr::name(&meta);
        match name.as_str() {
            "id" => id = attr::string(&meta)?,
            "formula" | "expr" => formula = attr::string(&meta)?,
            "unit" => unit = Some(attr::string(&meta)?),
            "deps" => attr::skip_string(&meta)?,
            _ => return Err(attr::unknown(&meta, &name)),
        }
        Ok(())
    });

    syn::parse::Parser::parse2(parser, args)?;

    let unit_expr = match unit {
        Some(u) => quote! { Some(#u) },
        None => quote! { None },
    };

    Ok(quote! {
        #input

        ::fluxbench::internal::inventory::submit! {
            ::fluxbench::SyntheticDef {
                id: #id,
                formula: #formula,
                unit: #unit_expr,
            }
        }
    })
}

/// Define a benchmark group
///
/// # Example
///
/// ```ignore
/// #[flux::group(
///     id = "parsing",
///     description = "Parser performance tests",
///     tags = "hot-path, critical"
/// )]
/// struct ParsingGroup;
///
/// // Nested groups
/// #[flux::group(
///     id = "json_parsing",
///     parent = "parsing",
///     description = "JSON-specific parsing benchmarks"
/// )]
/// struct JsonParsingGroup;
/// ```
#[proc_macro_attribute]
pub fn group(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = TokenStream2::from(args);
    let input = parse_macro_input!(item as ItemStruct);

    group_impl(args, input)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

fn group_impl(args: TokenStream2, input: ItemStruct) -> Result<TokenStream2, syn::Error> {
    let struct_name = &input.ident;
    let struct_name_str = struct_name.to_string();

    let mut id = struct_name_str.clone();
    let mut description = String::new();
    let mut tags: Vec<String> = Vec::new();
    let mut parent: Option<String> = None;

    let parser = syn::meta::parser(|meta| {
        let name = attr::name(&meta);
        match name.as_str() {
            "id" => id = attr::string(&meta)?,
            "description" | "desc" => description = attr::string(&meta)?,
            "tags" => tags = attr::tags(&meta)?,
            "parent" => parent = Some(attr::string(&meta)?),
            _ => return Err(attr::unknown(&meta, &name)),
        }
        Ok(())
    });

    syn::parse::Parser::parse2(parser, args)?;

    let tags_tokens: Vec<_> = tags.iter().map(|t| quote! { #t }).collect();
    let parent_expr = match parent {
        Some(p) => quote! { Some(#p) },
        None => quote! { None },
    };

    Ok(quote! {
        #input

        ::fluxbench::internal::inventory::submit! {
            ::fluxbench::GroupDef {
                id: #id,
                description: #description,
                tags: &[#(#tags_tokens),*],
                parent: #parent_expr,
            }
        }
    })
}

/// Define a dashboard report
///
/// # Example
///
/// ```ignore
/// #[flux::report(
///     title = "Performance Dashboard",
///     layout = "grid(2, 2)"
/// )]
/// struct MyDashboard;
/// ```
///
/// Note: Charts are added via separate ChartDef entries.
/// This macro sets up the dashboard structure.
#[proc_macro_attribute]
pub fn report(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = TokenStream2::from(args);
    let input = parse_macro_input!(item as ItemStruct);

    report_impl(args, input)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

fn report_impl(args: TokenStream2, input: ItemStruct) -> Result<TokenStream2, syn::Error> {
    let struct_name = &input.ident;
    let struct_name_str = struct_name.to_string();

    let mut title = struct_name_str.clone();
    let mut layout = (2u32, 2u32); // Default 2x2 grid

    let parser = syn::meta::parser(|meta| {
        let name = attr::name(&meta);
        match name.as_str() {
            "title" => title = attr::string(&meta)?,
            "layout" => layout = parse_grid_layout(&attr::string(&meta)?).unwrap_or((2, 2)),
            _ => return Err(attr::unknown(&meta, &name)),
        }
        Ok(())
    });

    syn::parse::Parser::parse2(parser, args)?;

    let (rows, cols) = layout;

    Ok(quote! {
        #input

        ::fluxbench::internal::inventory::submit! {
            ::fluxbench::ReportDef {
                title: #title,
                layout: (#rows, #cols),
                charts: &[], // Charts added separately
            }
        }
    })
}

/// Parse grid layout string like "grid(2, 2)" or "2x2"
fn parse_grid_layout(s: &str) -> Option<(u32, u32)> {
    let s = s.trim();

    // Try "grid(rows, cols)"
    if let Some(inner) = s.strip_prefix("grid(").and_then(|s| s.strip_suffix(')')) {
        let parts: Vec<&str> = inner.split(',').map(|p| p.trim()).collect();
        if parts.len() == 2 {
            if let (Ok(rows), Ok(cols)) = (parts[0].parse(), parts[1].parse()) {
                return Some((rows, cols));
            }
        }
    }

    // Try "rowsxcols" format
    if let Some(pos) = s.find('x') {
        let (rows_str, cols_str) = s.split_at(pos);
        let cols_str = &cols_str[1..]; // Skip 'x'
        if let (Ok(rows), Ok(cols)) = (rows_str.trim().parse(), cols_str.trim().parse()) {
            return Some((rows, cols));
        }
    }

    None
}

/// Define a comparison group for multiple benchmarks
///
/// # Example
///
/// ```ignore
/// #[flux::compare(
///     id = "parser_showdown",
///     title = "JSON Parser Comparison",
///     benchmarks = ["bench_serde", "bench_simd_json", "bench_sonic", "bench_jiter", "bench_yyjson"],
///     baseline = "bench_serde",
///     metric = "mean"
/// )]
/// struct ParserComparison;
/// ```
///
/// This creates a comparison group that will:
/// - Display all benchmarks in a comparison table
/// - Compute speedup ratios vs the baseline
/// - Generate data for comparison charts
#[proc_macro_attribute]
pub fn compare(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = TokenStream2::from(args);
    let input = parse_macro_input!(item as ItemStruct);

    compare_impl(args, input)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

fn compare_impl(args: TokenStream2, input: ItemStruct) -> Result<TokenStream2, syn::Error> {
    let struct_name = &input.ident;
    let struct_name_str = struct_name.to_string();

    let mut id = struct_name_str.clone();
    let mut title = struct_name_str.clone();
    let mut benchmarks: Vec<String> = Vec::new();
    let mut baseline: Option<String> = None;
    let mut metric = "mean".to_string();
    let mut group: Option<String> = None;
    let mut x_value: Option<String> = None;
    let mut series: Option<Vec<String>> = None;

    let parser = syn::meta::parser(|meta| {
        let name = attr::name(&meta);
        match name.as_str() {
            "id" => id = attr::string(&meta)?,
            "title" => title = attr::string(&meta)?,
            "benchmarks" => benchmarks = attr::string_array(&meta)?,
            "baseline" => baseline = Some(attr::string(&meta)?),
            "metric" => metric = attr::string(&meta)?,
            "group" => group = Some(attr::string(&meta)?),
            "x" => x_value = Some(attr::string(&meta)?),
            "series" => series = Some(attr::string_array(&meta)?),
            _ => return Err(attr::unknown(&meta, &name)),
        }
        Ok(())
    });

    syn::parse::Parser::parse2(parser, args)?;

    if benchmarks.is_empty() {
        return Err(syn::Error::new_spanned(
            &input,
            "compare requires at least one benchmark in 'benchmarks' list",
        ));
    }

    let baseline_expr = match &baseline {
        Some(b) => quote! { Some(#b) },
        None => quote! { None },
    };

    let group_expr = match &group {
        Some(g) => quote! { Some(#g) },
        None => quote! { None },
    };

    let x_expr = match &x_value {
        Some(x) => quote! { Some(#x) },
        None => quote! { None },
    };

    let series_expr = match &series {
        Some(s) => quote! { Some(&[#(#s),*]) },
        None => quote! { None },
    };

    Ok(quote! {
        #input

        ::fluxbench::internal::inventory::submit! {
            ::fluxbench::CompareDef {
                id: #id,
                title: #title,
                benchmarks: &[#(#benchmarks),*],
                baseline: #baseline_expr,
                metric: #metric,
                group: #group_expr,
                x: #x_expr,
                series: #series_expr,
            }
        }
    })
}
