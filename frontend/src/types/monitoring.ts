export type MonitoringSubject = {
  id: number;
  nome: string;
};

export type MonitoringClass = {
  id: number;
  nome: string;
  identificador?: string | null;
};

export type MonitoringOptions = {
  subjects: MonitoringSubject[];
  classes: MonitoringClass[];
  lesson_types: string[];
};

export type MonitoringSession = {
  id: number;
  status: string;
};

export type CreateMonitoringSessionPayload = {
  subject_id: number;
  class_id: number;
  lesson_type: string;
};
