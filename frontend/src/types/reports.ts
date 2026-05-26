export type ReportSummaryItem = {
  behavior: string;
  records: number;
  duration_seconds: number;
};

export type ReportSummaryResponse = {
  items: ReportSummaryItem[];
};

export type BehaviorEpisode = {
  id: number;
  monitoring_session_id?: number | null;
  student_id?: string | null;
  student: string;
  behavior: string;
  start_time: string;
  end_time: string;
  duration_seconds: number;
  date: string;
  source?: string | null;
  lesson_type?: string | null;
  discipline?: string | null;
  teacher?: string | null;
  class_name?: string | null;
  class_identifier?: string | null;
};

export type BehaviorEpisodeListResponse = {
  items: BehaviorEpisode[];
  total: number;
};
