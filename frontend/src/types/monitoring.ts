export type MonitoringSubject = {
  id: number;
  nome: string;
};

export type MonitoringClass = {
  id: number;
  nome: string;
  identificador?: string | null;
};

export type MonitoringAssignment = {
  subject_id: number;
  class_id: number;
};

export type MonitoringOptions = {
  subjects: MonitoringSubject[];
  classes: MonitoringClass[];
  assignments?: MonitoringAssignment[];
  lesson_types: string[];
};

export type MonitoringSession = {
  id: number;
  status: string;
};

export type MonitoringStatus = {
  status: string;
  session_id?: number | null;
  camera_status: string;
  websocket_clients: number;
  last_event?: string | null;
  error?: string | null;
  student_name: string;
  behavior: string;
  confidence: number;
  recognized_students_now?: number;
  timestamp?: string | null;
};

export type MonitoringStartPayload = {
  camera_id?: number | null;
  disciplina_id: number;
  turma_id: number;
  tipo_aula: string;
};

export type MonitoringRealtimeEvent = {
  type: string;
  session_id?: number | string | null;
  student_name?: string;
  behavior?: string;
  confidence?: number;
  faces_detected?: number;
  people_detected?: number;
  recognized_students_now?: number;
  camera_status?: string;
  websocket_status?: string;
  timestamp?: string;
  payload?: Record<string, unknown>;
};

export type CreateMonitoringSessionPayload = {
  subject_id: number;
  class_id: number;
  lesson_type: string;
};
