// Single-request bridge to official Codex client/auth libraries, pinned by native_client.py.
// No Codex Session, ThreadManager, agent loop, prompt assembly, or tool executor is constructed.
use codex_api::{
    ApiError, Compression, ReqwestTransport, ResponseEvent, ResponsesClient, TransportError,
};
use codex_http_client::{
    ClientRouteClass, HttpClientFactory, HttpTransport, OutboundProxyPolicy, Request,
};
use codex_login::{AuthCredentialsStoreMode, AuthKeyringBackendKind, AuthManager, AuthRouteConfig};
use codex_model_provider::auth_provider_from_auth_manager;
use codex_model_provider_info::{CHATGPT_CODEX_BASE_URL, ModelProviderInfo};
use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use std::fs::OpenOptions;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;
use tokio_tungstenite::tungstenite::{Message, client::IntoClientRequest};

fn emit(value: Value) {
    println!("{value}");
    let _ = io::stdout().flush();
}

async fn refinement_websocket(wire: &Request, body: &Value, request_id: &str) -> bool {
    // One-shot calls have no sessionId in Prime: no connection/context cache.
    let url = wire
        .url
        .replacen("https:", "wss:", 1)
        .replacen("http:", "ws:", 1);
    let Ok(mut handshake) = url.into_client_request() else {
        return false;
    };
    handshake.headers_mut().extend(wire.headers.clone());
    for name in ["accept", "content-type", "openai-beta"] {
        handshake.headers_mut().remove(name);
    }
    handshake.headers_mut().insert(
        "openai-beta",
        http::HeaderValue::from_static("responses_websockets=2026-02-06"),
    );
    let Ok(id) = http::HeaderValue::from_str(request_id) else {
        return false;
    };
    handshake
        .headers_mut()
        .insert("x-client-request-id", id.clone());
    handshake.headers_mut().insert("session_id", id);
    let Ok((mut socket, _)) = tokio_tungstenite::connect_async(handshake).await else {
        return false;
    };
    let mut request = body.clone();
    request["type"] = json!("response.create");
    if socket
        .send(Message::Text(request.to_string().into()))
        .await
        .is_err()
    {
        return false;
    }
    let mut started = false;
    let mut failure = "WebSocket stream closed before response.completed".to_owned();
    while let Some(frame) = socket.next().await {
        let text = match frame {
            Ok(Message::Text(text)) => text.to_string(),
            Ok(Message::Binary(bytes)) => {
                let decoded = String::from_utf8_lossy(&bytes);
                decoded
                    .strip_prefix('\u{feff}')
                    .unwrap_or(&decoded)
                    .to_owned()
            }
            Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => continue,
            Ok(Message::Close(frame)) => {
                failure = match frame {
                    Some(frame) => {
                        let reason = if frame.reason.is_empty() && u16::from(frame.code) == 1009 {
                            "message too big"
                        } else {
                            frame.reason.as_ref()
                        };
                        format!("WebSocket closed {} {}", u16::from(frame.code), reason)
                            .trim()
                            .to_owned()
                    }
                    None => "WebSocket closed 1005".to_owned(),
                };
                break;
            }
            Err(err) => {
                failure = err.to_string();
                break;
            }
            _ => break,
        };
        if text.is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(&text) {
            Ok(event) if !event.is_null() => {
                let kind = event.get("type").and_then(Value::as_str).unwrap_or("");
                let done = matches!(
                    kind,
                    "response.completed"
                        | "response.done"
                        | "response.incomplete"
                        | "response.failed"
                        | "error"
                );
                started |= !kind.is_empty();
                emit(json!({"type":"refinement_event", "event":event}));
                if done {
                    let _ = socket.close(None).await;
                    return true;
                }
            }
            _ => {
                emit(
                    json!({"type":"error", "code":"provider_failure", "error_message":"Invalid Codex WebSocket JSON", "retryable":true}),
                );
                return true; // Protocol errors never fall back, even before start.
            }
        }
    }
    if started {
        emit(
            json!({"type":"error", "code":"transport_failure", "error_message":failure, "retryable":true}),
        );
    }
    started // Transport failure before the first event permits SSE fallback.
}

fn error(error: &ApiError, refinement: bool) -> Value {
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
        if let ApiError::Transport(TransportError::Http {
            status,
            headers: source,
            body,
            ..
        }) = error
        {
            output["status"] = json!(status.as_u16());
            output["status_text"] = json!(status.canonical_reason().unwrap_or(""));
            // Only provider error data, never auth headers/request bodies. Prime
            // classifies the original provider code, not an SDK's replacement.
            output["provider_body"] = json!(body);
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
            ApiError::Retryable {
                delay: Some(delay), ..
            }
            | ApiError::RateLimitExceeded {
                delay: Some(delay), ..
            } => {
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
    if refinement {
        // Prime's one-shot path has no SDK retries or 401 recovery loop, and its
        // raw SSE parser preserves provider codes the SDK would otherwise lose.
        let transport = ReqwestTransport::from_http_client(http);
        let mut wire = Request::new(
            http::Method::POST,
            format!("{}/responses", provider.base_url.trim_end_matches('/')),
        )
        .with_json(&request["body"]);
        for (name, value) in [
            ("originator", "pi"),
            ("openai-beta", "responses=experimental"),
            ("accept", "text/event-stream"),
            ("content-type", "application/json"),
        ] {
            wire.headers.insert(
                http::header::HeaderName::from_static(name),
                http::HeaderValue::from_static(value),
            );
        }
        let auth_provider = auth_provider_from_auth_manager(Arc::clone(&manager), &auth);
        wire = auth_provider.apply_auth(wire).await?;
        wire.headers.insert(
            "user-agent",
            http::HeaderValue::from_str(
                request["user_agent"].as_str().ok_or("missing user agent")?,
            )?,
        );
        if refinement_websocket(
            &wire,
            &request["body"],
            request["request_id"].as_str().ok_or("missing request id")?,
        )
        .await
        {
            return Ok(());
        }
        match transport.stream(wire).await {
            Ok(mut response) => {
                while let Some(bytes) = response.bytes.next().await {
                    match bytes {
                        Ok(bytes) => emit(json!({"type":"refinement_sse", "bytes":bytes.to_vec()})),
                        Err(err) => {
                            emit(error(&ApiError::Transport(err), true));
                            return Ok(());
                        }
                    }
                }
                emit(json!({"type":"refinement_end"}));
            }
            Err(err) => emit(error(&ApiError::Transport(err), true)),
        }
        return Ok(());
    }
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

    #[tokio::test]
    async fn one_shot_websocket_fallback_and_protocol_boundaries() {
        for (frames, consumed) in [
            (vec![], false),
            (vec!["", "42", "[]"], false),
            (vec![r#"{"type":"response.created"}"#], true),
            (vec!["null"], true),
            (vec!["not json"], true),
            (vec![r#"{"type":"error","code":"invalid_api_key"}"#], true),
            (
                vec![r#"{"type":"response.completed","response":{"status":"completed"}}"#],
                true,
            ),
        ] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(async move {
                let (tcp, _) = listener.accept().await.unwrap();
                let mut socket = tokio_tungstenite::accept_hdr_async(
                    tcp,
                    |request: &tokio_tungstenite::tungstenite::handshake::server::Request,
                     response| {
                        assert_eq!(
                            request.headers()["openai-beta"],
                            "responses_websockets=2026-02-06"
                        );
                        assert_eq!(request.headers()["session_id"], "unit-request");
                        assert_eq!(request.headers()["x-client-request-id"], "unit-request");
                        assert_eq!(request.headers()["authorization"], "Bearer unit-fixture");
                        assert!(!request.headers().contains_key("accept"));
                        assert!(!request.headers().contains_key("content-type"));
                        Ok(response)
                    },
                )
                .await
                .unwrap();
                let message = socket.next().await.unwrap().unwrap().into_text().unwrap();
                let body: Value = serde_json::from_str(&message).unwrap();
                assert_eq!(body["type"], "response.create");
                assert_eq!(body["model"], "fixture");
                for frame in frames {
                    socket.send(Message::Text(frame.into())).await.unwrap();
                }
                let _ = socket.close(None).await;
            });
            let mut wire = Request::new(http::Method::POST, format!("http://{address}/responses"));
            for (key, value) in [
                ("authorization", "Bearer unit-fixture"),
                ("accept", "text/event-stream"),
                ("content-type", "application/json"),
            ] {
                wire.headers.insert(
                    http::header::HeaderName::from_static(key),
                    http::HeaderValue::from_static(value),
                );
            }
            let handled = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                refinement_websocket(&wire, &json!({"model":"fixture"}), "unit-request"),
            )
            .await
            .unwrap();
            assert_eq!(handled, consumed);
            server.await.unwrap();
        }
    }

    #[test]
    fn transport_errors_never_export_headers_or_response_bodies() {
        for (status, code, retryable) in [
            (401, "AUTH_REQUIRED", false),
            (429, "http_429", true),
            (500, "http_500", true),
        ] {
            let error = error(
                &ApiError::Transport(TransportError::Http {
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
                }),
                false,
            );
            assert_eq!(error["code"], code);
            assert_eq!(error["retryable"], retryable);
            assert!(!error.to_string().contains("SECRET"));
        }
    }
}
