from pydantic import BaseModel


class DashboardOverview(BaseModel):
    active_sessions: int
    students_monitored: int
    alerts: int
