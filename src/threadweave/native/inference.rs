// Single-request bridge to official Codex client/auth libraries, pinned by native_client.py.
// No Codex Session, ThreadManager, agent loop, prompt assembly, or tool executor is constructed.
use codex_api::{
    ApiError, Compression, ReqwestTransport, ResponseEvent, ResponsesClient, TransportError,
};
use codex_http_client::{ClientRouteClass, HttpClientFactory, OutboundProxyPolicy};
use codex_login::{AuthCredentialsStoreMode, AuthKeyringBackendKind, AuthManager, AuthRouteConfig};
use codex_model_provider::auth_provider_from_auth_manager;
use codex_model_provider_info::{CHATGPT_CODEX_BASE_URL, ModelProviderInfo};
use futures::StreamExt;
use serde_json::{Value, json};
use std::fs::OpenOptions;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;

fn emit(value: Value) {
    println!("{value}");
    let _ = io::stdout().flush();
}

fn error(error: &ApiError, refinement: bool) -> Value {
    if refinement {
        if let ApiError::Stream(message) = error {
            // The pinned Codex client represents response.incomplete as this
            // typed Stream error. Prime maps that event to stopReason=length.
            if message.starts_with("Incomplete response returned, reason: ") {
                return json!({"type":"completed", "stop_reason":"length"});
            }
        }
    }
    // Never stringify transport errors: their Display implementation includes HTTP bodies.
    let (code, retryable) = match error {
        ApiError::Transport(TransportError::Http { status, .. }) | ApiError::Api { status, .. } => {
            let status = status.as_u16();
            let code = if status == 401 {
                "AUTH_REQUIRED".into()
            } else {
                format!("http_{status}")
            };
            (code, status == 429 || status == 408 || status >= 500)
        }
        ApiError::ContextWindowExceeded => ("context_overflow".into(), false),
        ApiError::QuotaExceeded | ApiError::UsageNotIncluded => ("usage_limit".into(), false),
        ApiError::RateLimitExceeded { .. } | ApiError::RateLimit(_) => ("rate_limit".into(), true),
        ApiError::InvalidRequest { .. } => ("invalid_request".into(), false),
        ApiError::CyberPolicy { .. } | ApiError::MisalignmentPolicyViolation { .. } => {
            ("policy_restriction".into(), false)
        }
        _ => ("transport_failure".into(), true),
    };
    let mut output = json!({"type":"error", "code":code, "retryable":retryable});
    if refinement {
        let mut headers = serde_json::Map::new();
        if let ApiError::Transport(TransportError::Http { status, headers: source, .. }) = error {
            output["status"] = json!(status.as_u16());
            if let Some(source) = source {
                for name in ["retry-after", "retry-after-ms"] {
                    if let Some(value) = source.get(name).and_then(|v| v.to_str().ok()) {
                        headers.insert(name.into(), json!(value));
                    }
                }
            }
        } else if let ApiError::Api { status, .. } = error {
            output["status"] = json!(status.as_u16());
        }
        output["retry_headers"] = json!(headers);
        match error {
            ApiError::Retryable { delay: Some(delay), .. }
            | ApiError::RateLimitExceeded { delay: Some(delay), .. } => {
                output["retryAfterMs"] = json!(delay.as_secs_f64() * 1000.0);
            }
            _ => (),
        }
    }
    output
}

#[tokio::main]
async fn main() {
    if run().await.is_err() {
        emit(json!({"type":"error", "code":"client_setup", "retryable":false}));
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let line = io::stdin()
        .lock()
        .lines()
        .next()
        .ok_or("missing request")??;
    let request: Value = serde_json::from_str(&line)?;
    let refinement = request["refinement"] == true;
    let settings = &request["auth_settings"];
    let policy = if settings["respect_system_proxy"] == true {
        OutboundProxyPolicy::RespectSystemProxy
    } else {
        OutboundProxyPolicy::ReqwestDefault
    };
    let factory = HttpClientFactory::new(policy);
    let home = PathBuf::from(settings["codex_home"].as_str().ok_or("missing auth home")?);
    let store: AuthCredentialsStoreMode = serde_json::from_value(settings["store"].clone())?;
    let keyring: AuthKeyringBackendKind = serde_json::from_value(settings["keyring"].clone())?;
    let workspaces: Option<Vec<String>> = serde_json::from_value(settings["workspaces"].clone())?;
    // Coordinate refresh across Threadweave subprocesses without serializing inference.
    let lock = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(
            request["refresh_lock"]
                .as_str()
                .ok_or("missing refresh lock")?,
        )?;
    lock.lock()?;
    let manager = Arc::new(
        AuthManager::new(
            home,
            false,
            store,
            workspaces,
            None,
            keyring,
            AuthRouteConfig::from_http_client_factory(factory.clone()),
        )
        .await,
    );
    let auth = manager.auth().await.filter(|auth| auth.is_chatgpt_auth());
    lock.unlock()?;
    let Some(auth) = auth else {
        emit(json!({"type":"error", "code":"AUTH_REQUIRED", "retryable":false}));
        return Ok(());
    };
    if request["operation"] == "status" {
        emit(json!({"type":"status", "authentication":"chatgpt", "logged_in":true}));
        return Ok(());
    }
    let mut info = ModelProviderInfo::create_openai_provider(Some(CHATGPT_CODEX_BASE_URL.into()));
    info.env_http_headers = None; // No API organization/project or API endpoint fallback.
    info.request_max_retries = Some(0); // Threadweave owns inference retries and accounting.
    let provider = info.to_api_provider(Some(auth.auth_mode()))?;
    let http =
        factory.build_client_without_request_logging(&provider.base_url, ClientRouteClass::Api)?;
    let client = ResponsesClient::new(
        ReqwestTransport::from_http_client(http),
        provider,
        auth_provider_from_auth_manager(Arc::clone(&manager), &auth),
    );
    let mut recovery = manager.unauthorized_recovery();
    let mut request_headers = http::HeaderMap::new();
    if let Some(id) = request["request_id"].as_str() {
        let value = http::HeaderValue::from_str(id)?;
        request_headers.insert("x-client-request-id", value.clone());
        request_headers.insert("idempotency-key", value);
    }
    let stream = loop {
        match client
            .stream(
                request["body"].clone(),
                request_headers.clone(),
                Compression::None,
                None,
            )
            .await
        {
            Ok(stream) => break stream,
            Err(ApiError::Transport(TransportError::Http { status, .. }))
                if status.as_u16() == 401 && recovery.has_next() =>
            {
                lock.lock()?;
                let result = recovery.next().await;
                lock.unlock()?;
                emit(json!({"type":"auth_refresh", "succeeded":result.is_ok()}));
                if result.is_err() {
                    emit(json!({"type":"error", "code":"AUTH_REQUIRED", "retryable":false}));
                    return Ok(());
                }
            }
            Err(err) => {
                emit(error(&err, refinement));
                return Ok(());
            }
        }
    };
    let mut stream = stream;
    while let Some(event) = stream.next().await {
        match event {
            Ok(ResponseEvent::OutputTextDelta(text)) => {
                emit(json!({"type":"text_delta", "text":text}))
            }
            Ok(ResponseEvent::ToolCallInputDelta { call_id, delta, .. }) => {
                emit(json!({"type":"tool_delta", "call_id":call_id, "delta":delta}))
            }
            Ok(ResponseEvent::OutputItemDone(item)) => emit(json!({"type":"item", "item":item})),
            Ok(ResponseEvent::Completed {
                response_id,
                token_usage,
                end_turn,
                ..
            }) => {
                emit(
                    json!({"type":"completed", "id":response_id, "usage":token_usage, "end_turn":end_turn}),
                );
                return Ok(());
            }
            Ok(ResponseEvent::RateLimits(limits)) => {
                emit(json!({"type":"rate_limits", "limits":limits}))
            }
            Ok(ResponseEvent::ServerModel(model)) => emit(json!({"type":"model", "model":model})),
            Ok(ResponseEvent::ReasoningSummaryDelta { delta, .. }) => {
                emit(json!({"type":"reasoning_summary", "text":delta}))
            }
            Ok(_) => (),
            Err(err) => {
                emit(error(&err, refinement));
                return Ok(());
            }
        }
    }
    emit(json!({"type":"error", "code":"incomplete_stream", "retryable":true}));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_errors_never_export_headers_or_response_bodies() {
        for (status, code, retryable) in [
            (401, "AUTH_REQUIRED", false),
            (429, "http_429", true),
            (500, "http_500", true),
        ] {
            let error = error(&ApiError::Transport(TransportError::Http {
                status: http::StatusCode::from_u16(status).unwrap(),
                url: Some("https://private.example/SECRET".into()),
                headers: Some(
                    [(
                        http::header::AUTHORIZATION,
                        http::HeaderValue::from_static("Bearer SECRET"),
                    )]
                    .into_iter()
                    .collect(),
                ),
                body: Some("SECRET".into()),
            }));
            assert_eq!(error["code"], code);
            assert_eq!(error["retryable"], retryable);
            assert!(!error.to_string().contains("SECRET"));
        }
    }
}
