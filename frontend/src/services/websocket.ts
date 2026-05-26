const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL ?? "/api";

function resolveWebSocketBaseUrl() {
  if (WS_BASE_URL.startsWith("ws://") || WS_BASE_URL.startsWith("wss://")) {
    return WS_BASE_URL;
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${WS_BASE_URL}`;
}

export function createMonitoringSocket(token?: string): WebSocket {
  const url = new URL(`${resolveWebSocketBaseUrl()}/ws/monitoring`);
  if (token) {
    url.searchParams.set("token", token);
  }
  return new WebSocket(url);
}

export function getWebSocketBaseUrl() {
  return resolveWebSocketBaseUrl();
}
