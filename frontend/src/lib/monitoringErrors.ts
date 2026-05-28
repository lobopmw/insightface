const MONITORING_ERROR_MESSAGES: Record<string, string> = {
  opencv_not_installed:
    "OpenCV nao esta instalado no backend. Instale as dependencias de visao computacional e reinicie o servico.",
};

export function formatMonitoringError(error?: string | null) {
  if (!error) {
    return null;
  }

  return MONITORING_ERROR_MESSAGES[error] ?? error;
}
